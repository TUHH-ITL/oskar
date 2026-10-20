import pytest

from oskar_detection.detector import MODELS_DIR, load_spec, resolve_model


def test_every_model_file_parses():
    files = sorted(MODELS_DIR.glob("*.yaml"))
    assert files
    for path in files:
        spec = load_spec(path)
        assert spec.class_label
        assert spec.config.name.endswith(".yaml")


def test_bare_name_resolves_and_maskrcnn_config_defaults_next_to_weights():
    spec = load_spec("flowers_maskrcnn_sim")
    assert spec.architecture == "maskrcnn"
    assert spec.config == spec.weights.parent / "detectron2_config.yaml"


def test_unknown_model_is_an_error():
    with pytest.raises(FileNotFoundError):
        resolve_model("no_such_model")


def test_unknown_architecture_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("class_label: x\narchitecture: yolo\nweights: /nowhere.pth\n")
    with pytest.raises(ValueError):
        load_spec(bad)
