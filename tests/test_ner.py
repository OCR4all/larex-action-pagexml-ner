import pytest

from larex_action_pagexml_ner import main
from larex_action_pagexml_ner.main import NerPreloadState, get_ner_engine


def test_bundled_english_model_loads_and_filters_labels():
    pytest.importorskip("spacy")
    engine = get_ner_engine("en_core_web_sm")

    entities = engine.extract("Alice visited Berlin.", frozenset({"PERSON"}))

    assert [(entity.text, entity.label) for entity in entities] == [("Alice", "PERSON")]


class FakeEngine:
    model_name = "fake-model"

    def __init__(self):
        self.inputs: list[str] = []

    def extract(self, text, allowed_labels):
        self.inputs.append(text)
        return ()


@pytest.mark.asyncio
async def test_preload_loads_and_warms_model(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main, "get_ner_engine", lambda _model_name: engine)
    state = NerPreloadState.configured("fake-model")

    loaded = await main.preload_ner_engine(state)

    assert loaded is engine
    assert state.status == "ready"
    assert state.engine is engine
    assert engine.inputs == ["LAREX model warm-up."]


@pytest.mark.asyncio
async def test_load_ner_engine_awaits_matching_preload(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main, "get_ner_engine", lambda _model_name: engine)
    state = NerPreloadState.configured("fake-model")
    monkeypatch.setattr(main, "NER_PRELOAD_STATE", state)
    state.task = main.asyncio.create_task(main.preload_ner_engine(state))

    loaded = await main.load_ner_engine("fake-model")

    assert loaded is engine
    assert state.status == "ready"


def test_readiness_waits_for_model_and_reports_warmed_model():
    state = NerPreloadState.configured("fake-model")

    assert main.readiness_status(state, busy=False) == (
        503,
        {"status": "loading-model", "model": "fake-model"},
    )

    state.status = "ready"
    assert main.readiness_status(state, busy=False) == (
        200,
        {"status": "ready", "capacity": main.MAX_CONCURRENT_RUNS, "model": "fake-model"},
    )
    assert main.readiness_status(state, busy=True) == (503, {"status": "busy"})


def test_readiness_reports_preload_failure_without_exposing_error():
    state = NerPreloadState(
        model_name="fake-model",
        status="failed",
        error="secret filesystem detail",
    )

    assert main.readiness_status(state, busy=False) == (
        503,
        {"status": "model-unavailable", "model": "fake-model"},
    )
