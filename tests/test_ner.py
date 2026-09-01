import pytest

from larex_action_pagexml_ner import main
from larex_action_pagexml_ner.main import NerPreloadState, discover_spacy_models, get_ner_engine


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


def test_discovers_installed_and_nested_mounted_models(monkeypatch, tmp_path):
    spacy = pytest.importorskip("spacy")
    monkeypatch.setattr(spacy.util, "get_installed_models", lambda: ["installed_model"])
    model_dir = tmp_path / "training-run" / "model-best"
    model_dir.mkdir(parents=True)
    (model_dir / "config.cfg").write_text("[nlp]\n", encoding="utf-8")

    choices = discover_spacy_models(tmp_path)

    values = {choice.value: choice.label for choice in choices}
    assert values["installed_model"] == "installed model"
    assert values[str(model_dir.resolve())] == "training-run/model-best"
    assert [choice.label for choice in choices] == sorted(
        [choice.label for choice in choices], key=str.casefold
    )


def test_model_discovery_does_not_follow_paths_outside_configured_root(monkeypatch, tmp_path):
    spacy = pytest.importorskip("spacy")
    monkeypatch.setattr(spacy.util, "get_installed_models", lambda: [])
    root = tmp_path / "models"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "config.cfg").write_text("[nlp]\n", encoding="utf-8")
    (root / "escaped").symlink_to(outside, target_is_directory=True)

    values = {choice.value for choice in discover_spacy_models(root)}

    assert str(outside.resolve()) not in values
