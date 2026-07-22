from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from larex_actions import ResultBuilder

from larex_action_pagexml_ner import main
from larex_action_pagexml_ner.main import Entity

PAGE_XML = b"""<PcGts xmlns="urn:page"><Page>
  <TextRegion id="region-1">
    <TextLine id="line-1"><TextEquiv><Unicode>Alice visits Berlin.</Unicode></TextEquiv></TextLine>
  </TextRegion>
</Page></PcGts>"""


class FakeNer:
    model_name = "fake-ner"

    def extract(self, text, allowed_labels):
        entities = (Entity("Alice", "PERSON", 0, 5), Entity("Berlin", "GPE", 13, 19))
        return tuple(
            entity for entity in entities if not allowed_labels or entity.label in allowed_labels
        )


class FakeContext:
    def __init__(self, action_input):
        self.input = action_input
        self.events: list[tuple[str, object]] = []
        self.uploads: dict[str, bytes] = {}
        self.result_manifest: dict | None = None

    async def pull_input(self):
        return self.input

    async def check_cancelled(self):
        self.events.append(("cancel", None))

    async def heartbeat(self, progress, message, *, raise_on_cancel=False):
        self.events.append(("heartbeat", progress))

    async def download_bytes(self, file):
        return file.content

    def result_builder(self):
        return ResultBuilder()

    async def complete(self, results=None, message=None):
        self.events.append(("complete", message))
        if results is None:
            return
        with ExitStack() as exit_stack:
            parts = results.httpx_files(status="completed", message=message, exit_stack=exit_stack)
            for field_name, (file_name, content, _mime_type) in parts:
                raw = content.read() if hasattr(content, "read") else content
                if isinstance(raw, str):
                    raw = raw.encode()
                self.uploads[file_name] = raw
                if field_name == "manifest":
                    import json

                    self.result_manifest = json.loads(raw)


def action_input(*, custom_files=True, parameters=None, pages=None):
    pages = pages if pages is not None else [page("page-1", "Page One")]
    return SimpleNamespace(
        run_id="run-1",
        project_id="project-1",
        parameters=parameters or {},
        capabilities=SimpleNamespace(custom_file_results=custom_files),
        pages=pages,
    )


def page(page_id: str, name: str, xml: bytes = PAGE_XML):
    xml_file = SimpleNamespace(file_size=len(xml), content=xml)
    return SimpleNamespace(id=page_id, name=name, xml=[xml_file])


@pytest.mark.asyncio
async def test_process_run_uploads_page_and_project_files_atomically(monkeypatch):
    context = FakeContext(
        action_input(
            parameters={
                "enableNer": True,
                "nerModel": "fake-ner",
                "entityLabels": "PERSON,GPE",
            },
            pages=[page("page-1", "Same name"), page("page-2", "Same name")],
        )
    )
    monkeypatch.setattr(main, "get_ner_engine", lambda _model_name: FakeNer())

    await main.process_run(context)

    assert set(context.uploads) == {
        "manifest.json",
        "0001-Same-name.txt",
        "0001-Same-name.entities.json",
        "0002-Same-name.txt",
        "0002-Same-name.entities.json",
        "document.txt",
        "entities.csv",
    }
    assert (
        context.uploads["document.txt"]
        == PAGE_XML_TEXT + main.DOCUMENT_SEPARATOR.encode() + PAGE_XML_TEXT
    )
    assert context.result_manifest is not None
    result_files = context.result_manifest["files"]
    assert len(result_files) == 7
    assert [file["pageId"] for file in result_files[:4]] == [
        "page-1",
        "page-1",
        "page-2",
        "page-2",
    ]
    assert all(event[0] != "submit" for event in context.events)
    assert context.events[-1] == (
        "complete",
        "Exported 2 page(s) and 4 named entity occurrence(s).",
    )


PAGE_XML_TEXT = b"Alice visits Berlin."


@pytest.mark.asyncio
async def test_process_run_without_pages_creates_no_output():
    context = FakeContext(action_input(pages=[]))

    await main.process_run(context)

    assert context.uploads == {}
    assert context.events == [
        ("complete", "PAGE XML export received no pages; no output was created.")
    ]


@pytest.mark.asyncio
async def test_process_run_requires_custom_file_capability():
    context = FakeContext(action_input(custom_files=False))

    with pytest.raises(RuntimeError, match="customFileResults"):
        await main.process_run(context)

    assert not any(event[0] == "complete" for event in context.events)


@pytest.mark.asyncio
async def test_process_run_requires_model_when_ner_is_enabled():
    context = FakeContext(action_input(parameters={"enableNer": True, "nerModel": ""}))

    with pytest.raises(ValueError, match="nerModel"):
        await main.process_run(context)


@pytest.mark.asyncio
async def test_continue_on_invalid_xml_records_warning_and_exports_valid_pages():
    context = FakeContext(
        action_input(
            parameters={"continueOnInvalidXml": True},
            pages=[page("bad", "Bad", b"<broken"), page("good", "Good")],
        )
    )

    await main.process_run(context)

    import json

    manifest = json.loads(context.uploads["manifest.json"])
    assert manifest["warnings"][0]["pageId"] == "bad"
    assert set(context.uploads) == {
        "manifest.json",
        "0001-Good.txt",
        "0001-Good.entities.json",
        "document.txt",
        "entities.csv",
    }
    assert context.events[-1] == (
        "complete",
        "Exported 1 page(s) and 0 named entity occurrence(s). Skipped 1 invalid page(s).",
    )
