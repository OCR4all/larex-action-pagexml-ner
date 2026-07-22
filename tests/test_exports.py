import csv
import json
from io import StringIO
from types import SimpleNamespace

from larex_action_pagexml_ner.main import (
    Entity,
    ExportParameters,
    OutputMetadata,
    PageExport,
    page_entity_json,
    project_entities_csv,
    project_manifest_json,
)


class FakeNer:
    model_name = "test-model"


def page_export() -> PageExport:
    return PageExport(
        page_id="page-1",
        page_name="Page One",
        text="Alice visits W\u00fcrzburg.",
        entities=(
            Entity("Alice", "PERSON", 0, 5),
            Entity("W\u00fcrzburg", "GPE", 13, 22, "Q2999"),
        ),
        region_count=1,
        line_count=1,
        document_start=10,
        document_end=33,
    )


def test_page_entity_json_uses_page_offsets():
    payload = json.loads(page_entity_json(page_export(), FakeNer()))

    assert payload["model"] == "test-model"
    assert payload["entities"][0] == {
        "text": "Alice",
        "label": "PERSON",
        "start": 0,
        "end": 5,
        "kbId": None,
    }


def test_project_entities_csv_adds_document_offsets_and_quotes_values():
    rows = list(csv.DictReader(StringIO(project_entities_csv([page_export()]))))

    assert rows[0]["document_start"] == "10"
    assert rows[0]["document_end"] == "15"
    assert rows[1]["kb_id"] == "Q2999"


def test_manifest_contains_provenance_configuration_and_checksums():
    action_input = SimpleNamespace(run_id="run-1", project_id="project-1")
    output = OutputMetadata("page.txt", "text/plain", 3, "abc", "page-1")

    payload = json.loads(
        project_manifest_json(
            action_input=action_input,
            parameters=ExportParameters(enableNer=False),
            exports=[page_export()],
            warnings=[],
            outputs=[output],
            ner_engine=None,
        )
    )

    assert payload["formatVersion"] == 1
    assert payload["source"] == {"runId": "run-1", "projectId": "project-1", "pageCount": 1}
    assert payload["files"][0]["sha256"] == "abc"
    assert payload["pages"][0]["documentStart"] == 10


def test_parameter_labels_are_trimmed_and_case_normalized():
    parameters = ExportParameters(entityLabels=" person, Gpe ,,DATE ")

    assert parameters.allowed_entity_labels == frozenset({"PERSON", "GPE", "DATE"})
