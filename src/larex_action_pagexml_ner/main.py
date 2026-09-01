from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from larex_actions import ActionContext, ParameterChoice
from larex_actions.fastapi import create_larex_action_app
from lxml import etree
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__


def configure_sdk_transport_logging() -> None:
    enabled = os.getenv("LAREX_SDK_TRANSPORT_LOGGING", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    sdk_logger = logging.getLogger("larex_actions.transport")
    sdk_logger.setLevel(logging.DEBUG)
    if not sdk_logger.hasHandlers():
        sdk_logger.addHandler(logging.StreamHandler())
        sdk_logger.propagate = False


configure_sdk_transport_logging()


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.")
    return value


PROCESSOR_ID = os.getenv("LAREX_PROCESSOR_ID", "pagexml-text-ner-export")
DISPATCH_SECRET_ENV = "LAREX_DISPATCH_HMAC_SECRET"
PRELOAD_NER_MODEL = os.getenv("LAREX_PRELOAD_NER_MODEL", "en_core_web_sm").strip()
NER_MODEL_DIRECTORY = os.getenv("LAREX_NER_MODEL_DIRECTORY", "").strip()
MAX_CONCURRENT_RUNS = positive_int_env("LAREX_MAX_CONCURRENT_RUNS", 1)
MAX_XML_BYTES = positive_int_env("LAREX_MAX_XML_BYTES", 50 * 1024 * 1024)
RUN_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
PROGRESS_COMPLETE_BEFORE_UPLOAD = 95
DOCUMENT_SEPARATOR = "\n\n\f\n\n"
DEHYPHENATION_MARKS = "-\u00ad\u2010\u2011"
logger = logging.getLogger(__name__)


class ExportParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    preserve_line_breaks: bool = Field(default=True, alias="preserveLineBreaks")
    dehyphenate: bool = True
    unicode_normalization: Literal["none", "NFC", "NFKC"] = Field(
        default="NFC", alias="unicodeNormalization"
    )
    enable_ner: bool = Field(default=False, alias="enableNer")
    ner_model: str = Field(default="", alias="nerModel", max_length=1024)
    entity_labels: str = Field(default="", alias="entityLabels", max_length=1024)
    continue_on_invalid_xml: bool = Field(default=False, alias="continueOnInvalidXml")

    @field_validator("ner_model", "entity_labels")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        return value.strip()

    @property
    def allowed_entity_labels(self) -> frozenset[str]:
        return frozenset(
            label.strip().upper() for label in self.entity_labels.split(",") if label.strip()
        )


@dataclass(frozen=True)
class ExtractedPageText:
    text: str
    region_count: int
    line_count: int


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
    kb_id: str | None = None


@dataclass(frozen=True)
class PageExport:
    page_id: str
    page_name: str
    text: str
    entities: tuple[Entity, ...]
    region_count: int
    line_count: int
    document_start: int
    document_end: int


@dataclass(frozen=True)
class OutputMetadata:
    file_name: str
    mime_type: str
    size: int
    sha256: str
    page_id: str | None = None


class NerEngine(Protocol):
    model_name: str

    def extract(self, text: str, allowed_labels: frozenset[str]) -> tuple[Entity, ...]: ...


@dataclass
class NerPreloadState:
    model_name: str
    status: Literal["disabled", "pending", "loading", "ready", "failed"]
    engine: NerEngine | None = None
    error: str | None = None
    task: asyncio.Task[NerEngine | None] | None = None

    @classmethod
    def configured(cls, model_name: str) -> NerPreloadState:
        return cls(model_name=model_name, status="pending" if model_name else "disabled")


class SpacyNerEngine:
    def __init__(self, model_name: str) -> None:
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "NER is enabled but spaCy is not installed. "
                "Install the processor with the 'ner' extra."
            ) from exc
        try:
            self.pipeline = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load spaCy model {model_name!r}. "
                "Install it in the processor image or mount it as a path."
            ) from exc
        self.model_name = model_name

    def extract(self, text: str, allowed_labels: frozenset[str]) -> tuple[Entity, ...]:
        document = self.pipeline(text)
        return tuple(
            Entity(
                text=entity.text,
                label=entity.label_,
                start=entity.start_char,
                end=entity.end_char,
                kb_id=entity.kb_id_ or None,
            )
            for entity in document.ents
            if not allowed_labels or entity.label_.upper() in allowed_labels
        )


@lru_cache(maxsize=4)
def get_ner_engine(model_name: str) -> NerEngine:
    return SpacyNerEngine(model_name)


def discover_spacy_models(
    model_directory: str | Path | None = NER_MODEL_DIRECTORY,
) -> list[ParameterChoice]:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required to discover NER models") from exc

    choices: dict[str, ParameterChoice] = {}
    for model_name in spacy.util.get_installed_models():
        choices[model_name] = ParameterChoice(
            value=model_name,
            label=model_name.replace("_", " "),
        )
    if PRELOAD_NER_MODEL:
        choices.setdefault(
            PRELOAD_NER_MODEL,
            ParameterChoice(
                value=PRELOAD_NER_MODEL,
                label=Path(PRELOAD_NER_MODEL).name.replace("_", " "),
            ),
        )

    if model_directory:
        root = Path(model_directory).expanduser().resolve()
        if root.is_dir():
            for config_path in root.glob("**/config.cfg"):
                model_path = config_path.parent.resolve()
                if not model_path.is_relative_to(root):
                    continue
                value = str(model_path)
                label = str(model_path.relative_to(root))
                choices[value] = ParameterChoice(value=value, label=label)
                if len(choices) >= 1_000:
                    break

    return sorted(choices.values(), key=lambda choice: (choice.label.casefold(), str(choice.value)))


NER_PRELOAD_STATE = NerPreloadState.configured(PRELOAD_NER_MODEL)


async def preload_ner_engine(state: NerPreloadState) -> NerEngine | None:
    if not state.model_name:
        state.status = "disabled"
        return None
    state.status = "loading"
    state.error = None
    try:
        engine = await asyncio.to_thread(get_ner_engine, state.model_name)
        await asyncio.to_thread(engine.extract, "LAREX model warm-up.", frozenset())
    except Exception as exc:
        state.status = "failed"
        state.error = f"{exc.__class__.__name__}: {exc}"
        logger.exception("Could not preload spaCy model %s", state.model_name)
        return None
    state.engine = engine
    state.status = "ready"
    logger.info("Preloaded spaCy model %s", state.model_name)
    return engine


async def load_ner_engine(model_name: str) -> NerEngine:
    if model_name == NER_PRELOAD_STATE.model_name and NER_PRELOAD_STATE.task is not None:
        engine = await asyncio.shield(NER_PRELOAD_STATE.task)
        if engine is None:
            raise RuntimeError(f"Preloaded spaCy model {model_name!r} is unavailable.")
        return engine
    return await asyncio.to_thread(get_ner_engine, model_name)


@asynccontextmanager
async def application_lifespan(_app: FastAPI):
    if NER_PRELOAD_STATE.model_name:
        NER_PRELOAD_STATE.status = "pending"
        NER_PRELOAD_STATE.engine = None
        NER_PRELOAD_STATE.error = None
        NER_PRELOAD_STATE.task = asyncio.create_task(preload_ner_engine(NER_PRELOAD_STATE))
    else:
        NER_PRELOAD_STATE.status = "disabled"
    try:
        yield
    finally:
        task = NER_PRELOAD_STATE.task
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        NER_PRELOAD_STATE.task = None


async def process_run(ctx: ActionContext) -> None:
    async with RUN_SEMAPHORE:
        await _process_run(ctx)


async def _process_run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.complete(message="PAGE XML export received no pages; no output was created.")
        return
    if not action_input.capabilities.custom_file_results:
        raise RuntimeError("This processor requires LAREX customFileResults support.")

    parameters = ExportParameters.model_validate(action_input.parameters)
    if parameters.enable_ner and not parameters.ner_model:
        raise ValueError("nerModel must be configured when enableNer is true.")
    ner_engine = await load_ner_engine(parameters.ner_model) if parameters.enable_ner else None

    exports: list[PageExport] = []
    warnings: list[dict[str, str]] = []
    output_metadata: list[OutputMetadata] = []
    document_parts: list[str] = []
    document_length = 0

    with tempfile.TemporaryDirectory(prefix="larex-pagexml-ner-") as temp_dir:
        work_dir = Path(temp_dir)
        results = ctx.result_builder()

        for index, page in enumerate(action_input.pages, start=1):
            await ctx.check_cancelled()
            await ctx.heartbeat(
                page_progress(index - 1, len(action_input.pages)),
                f"Exporting page {index}/{len(action_input.pages)}: {page.name}",
                raise_on_cancel=True,
            )
            try:
                page_text = await download_and_extract_page(ctx, page, parameters)
            except (ValueError, etree.XMLSyntaxError) as exc:
                if not parameters.continue_on_invalid_xml:
                    raise
                warnings.append({"pageId": page.id, "pageName": page.name, "message": str(exc)})
                continue

            entities = (
                await asyncio.to_thread(
                    ner_engine.extract, page_text.text, parameters.allowed_entity_labels
                )
                if ner_engine is not None
                else ()
            )
            if document_parts:
                document_length += len(DOCUMENT_SEPARATOR)
            document_start = document_length
            document_parts.append(page_text.text)
            document_length += len(page_text.text)

            page_export = PageExport(
                page_id=page.id,
                page_name=page.name,
                text=page_text.text,
                entities=entities,
                region_count=page_text.region_count,
                line_count=page_text.line_count,
                document_start=document_start,
                document_end=document_length,
            )
            exports.append(page_export)
            await ctx.heartbeat(
                page_progress(index, len(action_input.pages)),
                f"Prepared page {index}/{len(action_input.pages)}: {page.name}",
                raise_on_cancel=True,
            )

        if not exports:
            await ctx.complete(
                message=f"No valid PAGE XML was exported ({len(warnings)} page warning(s))."
            )
            return

        for index, export in enumerate(exports, start=1):
            prefix = numbered_page_stem(index, export.page_name, export.page_id)
            text_path = work_dir / f"{prefix}.txt"
            entity_path = work_dir / f"{prefix}.entities.json"
            text_path.write_text(export.text, encoding="utf-8")
            entity_path.write_text(page_entity_json(export, ner_engine), encoding="utf-8")
            add_output_path(
                results,
                output_metadata,
                text_path,
                mime_type="text/plain; charset=utf-8",
                page_id=export.page_id,
            )
            add_output_path(
                results,
                output_metadata,
                entity_path,
                mime_type="application/json",
                page_id=export.page_id,
            )

        document_path = work_dir / "document.txt"
        entities_path = work_dir / "entities.csv"
        document_path.write_text(DOCUMENT_SEPARATOR.join(document_parts), encoding="utf-8")
        entities_path.write_text(project_entities_csv(exports), encoding="utf-8", newline="")
        add_output_path(
            results, output_metadata, document_path, mime_type="text/plain; charset=utf-8"
        )
        add_output_path(
            results, output_metadata, entities_path, mime_type="text/csv; charset=utf-8"
        )

        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(
            project_manifest_json(
                action_input=action_input,
                parameters=parameters,
                exports=exports,
                warnings=warnings,
                outputs=output_metadata,
                ner_engine=ner_engine,
            ),
            encoding="utf-8",
        )
        add_output_path(results, output_metadata, manifest_path, mime_type="application/json")

        await ctx.complete(
            results,
            result_message(
                page_count=len(exports),
                entity_count=sum(len(export.entities) for export in exports),
                warning_count=len(warnings),
            ),
        )


async def download_and_extract_page(
    ctx: ActionContext, page, parameters: ExportParameters
) -> ExtractedPageText:
    if not page.xml:
        raise ValueError(f"Page {page.id} does not expose a PAGE XML input.")
    xml_file = page.xml[0]
    if xml_file.file_size is not None and xml_file.file_size > MAX_XML_BYTES:
        raise ValueError(
            f"PAGE XML for page {page.id} exceeds the {MAX_XML_BYTES}-byte processor limit."
        )
    xml_bytes = await ctx.download_bytes(xml_file)
    if len(xml_bytes) > MAX_XML_BYTES:
        raise ValueError(
            f"PAGE XML for page {page.id} exceeds the {MAX_XML_BYTES}-byte processor limit."
        )
    return extract_page_text(
        xml_bytes,
        preserve_line_breaks=parameters.preserve_line_breaks,
        dehyphenate=parameters.dehyphenate,
        unicode_normalization=parameters.unicode_normalization,
    )


def extract_page_text(
    xml_bytes: bytes,
    *,
    preserve_line_breaks: bool = True,
    dehyphenate: bool = True,
    unicode_normalization: Literal["none", "NFC", "NFKC"] = "NFC",
) -> ExtractedPageText:
    root = parse_xml(xml_bytes)
    page = next((element for element in root.iter() if local_name(element.tag) == "Page"), None)
    if page is None:
        raise ValueError("Input does not contain a PAGE XML Page element.")

    regions = [element for element in page.iter() if local_name(element.tag) == "TextRegion"]
    regions_by_id = {region.get("id"): region for region in regions if region.get("id")}
    ordered_region_ids = page_reading_order(page)
    ordered_regions = [
        regions_by_id[region_id] for region_id in ordered_region_ids if region_id in regions_by_id
    ]
    referenced = {region.get("id") for region in ordered_regions}
    ordered_regions.extend(region for region in regions if region.get("id") not in referenced)

    lines: list[str] = []
    line_count = 0
    for region in ordered_regions:
        region_lines = [
            element for element in region.iter() if local_name(element.tag) == "TextLine"
        ]
        values = [value for line in region_lines if (value := text_equiv_unicode(line)) is not None]
        if values:
            lines.extend(values)
            line_count += len(values)
            continue
        region_value = text_equiv_unicode(region)
        if region_value is not None:
            lines.append(region_value)
            line_count += 1

    text = combine_lines(lines, preserve_line_breaks=preserve_line_breaks, dehyphenate=dehyphenate)
    if unicode_normalization != "none":
        text = unicodedata.normalize(unicode_normalization, text)
    return ExtractedPageText(text=text, region_count=len(regions), line_count=line_count)


def page_reading_order(page: etree._Element) -> list[str]:
    reading_order = next((child for child in page if local_name(child.tag) == "ReadingOrder"), None)
    if reading_order is None:
        return []
    result: list[str] = []

    def walk(container: etree._Element) -> None:
        children = [
            child
            for child in container
            if "RegionRef" in local_name(child.tag) or "Group" in local_name(child.tag)
        ]
        indexed = sorted(
            enumerate(children),
            key=lambda item: (parse_index(item[1].get("index")), item[0]),
        )
        for _, child in indexed:
            region_ref = child.get("regionRef")
            if region_ref:
                if region_ref not in result:
                    result.append(region_ref)
            else:
                walk(child)

    walk(reading_order)
    return result


def parse_index(value: str | None) -> int:
    if value is None:
        return 2**31 - 1
    try:
        return int(value)
    except ValueError:
        return 2**31 - 1


def text_equiv_unicode(element: etree._Element) -> str | None:
    text_equivs = [child for child in element if local_name(child.tag) == "TextEquiv"]
    if not text_equivs:
        return None
    selected = min(
        enumerate(text_equivs),
        key=lambda item: (parse_index(item[1].get("index")), item[0]),
    )[1]
    unicode_element = next(
        (child for child in selected if local_name(child.tag) == "Unicode"), None
    )
    if unicode_element is None:
        return None
    return "".join(unicode_element.itertext()).strip()


def combine_lines(lines: Sequence[str], *, preserve_line_breaks: bool, dehyphenate: bool) -> str:
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        return ""
    result = nonempty[0]
    separator = "\n" if preserve_line_breaks else " "
    for line in nonempty[1:]:
        if dehyphenate and should_join_hyphenated(result, line):
            result = result[:-1] + line.lstrip()
        else:
            result += separator + line
    return result


def should_join_hyphenated(previous: str, following: str) -> bool:
    return bool(
        previous
        and following
        and previous[-1] in DEHYPHENATION_MARKS
        and following.lstrip()[:1].islower()
    )


def page_entity_json(export: PageExport, ner_engine: NerEngine | None) -> str:
    return (
        json.dumps(
            {
                "pageId": export.page_id,
                "pageName": export.page_name,
                "model": ner_engine.model_name if ner_engine is not None else None,
                "textLength": len(export.text),
                "entities": [entity_json(entity) for entity in export.entities],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def entity_json(entity: Entity) -> dict[str, str | int | None]:
    return {
        "text": entity.text,
        "label": entity.label,
        "start": entity.start,
        "end": entity.end,
        "kbId": entity.kb_id,
    }


def project_entities_csv(exports: Iterable[PageExport]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "page_id",
            "page_name",
            "label",
            "text",
            "start",
            "end",
            "document_start",
            "document_end",
            "kb_id",
        ],
    )
    writer.writeheader()
    for export in exports:
        for entity in export.entities:
            writer.writerow(
                {
                    "page_id": export.page_id,
                    "page_name": export.page_name,
                    "label": entity.label,
                    "text": entity.text,
                    "start": entity.start,
                    "end": entity.end,
                    "document_start": export.document_start + entity.start,
                    "document_end": export.document_start + entity.end,
                    "kb_id": entity.kb_id or "",
                }
            )
    return output.getvalue()


def project_manifest_json(
    *,
    action_input,
    parameters: ExportParameters,
    exports: Sequence[PageExport],
    warnings: Sequence[dict[str, str]],
    outputs: Sequence[OutputMetadata],
    ner_engine: NerEngine | None,
) -> str:
    return (
        json.dumps(
            {
                "format": "larex-pagexml-text-ner-export",
                "formatVersion": 1,
                "processor": {"id": PROCESSOR_ID, "version": __version__},
                "source": {
                    "runId": action_input.run_id,
                    "projectId": action_input.project_id,
                    "pageCount": len(exports),
                },
                "configuration": parameters.model_dump(by_alias=True),
                "ner": {
                    "enabled": ner_engine is not None,
                    "model": ner_engine.model_name if ner_engine is not None else None,
                },
                "pages": [
                    {
                        "pageId": export.page_id,
                        "pageName": export.page_name,
                        "regionCount": export.region_count,
                        "lineCount": export.line_count,
                        "textLength": len(export.text),
                        "entityCount": len(export.entities),
                        "documentStart": export.document_start,
                        "documentEnd": export.document_end,
                    }
                    for export in exports
                ],
                "warnings": list(warnings),
                "files": [output_json(output) for output in outputs],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def output_json(output: OutputMetadata) -> dict[str, str | int | None]:
    return {
        "fileName": output.file_name,
        "mimeType": output.mime_type,
        "size": output.size,
        "sha256": output.sha256,
        "pageId": output.page_id,
    }


def add_output_path(
    results,
    metadata: list[OutputMetadata],
    path: Path,
    *,
    mime_type: str,
    page_id: str | None = None,
) -> None:
    results.add_file_path(path, file_name=path.name, mime_type=mime_type, page_id=page_id)
    content = path.read_bytes()
    metadata.append(
        OutputMetadata(
            file_name=path.name,
            mime_type=mime_type,
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            page_id=page_id,
        )
    )


def numbered_page_stem(index: int, page_name: str, page_id: str) -> str:
    return f"{index:04d}-{safe_stem(page_name or page_id)}"


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value).stem).strip("._-")
    return stem[:96] or "page"


def page_progress(completed_pages: int, total_pages: int) -> int:
    return int((completed_pages / total_pages) * PROGRESS_COMPLETE_BEFORE_UPLOAD)


def result_message(page_count: int, entity_count: int, warning_count: int) -> str:
    message = f"Exported {page_count} page(s) and {entity_count} named entity occurrence(s)."
    if warning_count:
        message += f" Skipped {warning_count} invalid page(s)."
    return message


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def parse_xml(xml_bytes: bytes) -> etree._Element:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    return etree.fromstring(xml_bytes, parser=parser)


fastapi_app = FastAPI(
    title=f"LAREX Action Processor: {PROCESSOR_ID}",
    lifespan=application_lifespan,
)
app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret_env=DISPATCH_SECRET_ENV,
    handler=process_run,
    app=fastapi_app,
    parameter_value_providers={"spacyModels": discover_spacy_models},
)


def readiness_status(state: NerPreloadState, *, busy: bool) -> tuple[int, dict[str, str | int]]:
    if state.status in {"pending", "loading"}:
        return 503, {"status": "loading-model", "model": state.model_name}
    if state.status == "failed":
        return 503, {"status": "model-unavailable", "model": state.model_name}
    if busy:
        return 503, {"status": "busy"}
    payload: dict[str, str | int] = {
        "status": "ready",
        "capacity": MAX_CONCURRENT_RUNS,
    }
    if state.status == "ready":
        payload["model"] = state.model_name
    return 200, payload


def configured_readiness_paths() -> frozenset[str]:
    paths = {"/ready"}
    raw_prefixes = os.getenv("LAREX_ACTION_ROUTE_PREFIXES", "")
    for raw_prefix in raw_prefixes.split(","):
        prefix = raw_prefix.strip().strip("/")
        if prefix:
            paths.add(f"/{prefix}/ready")
    return frozenset(paths)


READINESS_PATHS = configured_readiness_paths()


@app.middleware("http")
async def model_readiness(request: Request, call_next):
    if request.method == "GET" and request.url.path in READINESS_PATHS:
        status_code, payload = readiness_status(
            NER_PRELOAD_STATE,
            busy=RUN_SEMAPHORE.locked(),
        )
        return JSONResponse(payload, status_code=status_code)

    return await call_next(request)
