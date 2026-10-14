"""The shipped workflows against a trimmed /object_info snapshot of a working ComfyUI (0.30.0, KJNodes 1.5)."""

import json
from pathlib import Path

import pytest

from gigapixel_upscale import graph

ROOT = Path(__file__).parent.parent
OBJECT_INFO = json.loads((Path(__file__).parent / "fixtures" / "object_info.json").read_text())
UI = json.loads((ROOT / "workflows" / "gigapixel_upscale.json").read_text())
API = graph.load()
LOADERS = {"UpscaleModelLoader", "UNETLoader", "VAELoader"}


def declared_inputs(class_type):
    spec = OBJECT_INFO[class_type]["input"]
    return spec.get("required", {}), spec.get("optional", {})


@pytest.mark.parametrize("nid", sorted(API, key=int))
def test_every_node_exists_and_only_uses_declared_inputs(nid):
    node = API[nid]
    required, optional = declared_inputs(node["class_type"])
    assert set(node["inputs"]) <= set(required) | set(optional)
    assert set(required) <= set(node["inputs"])


def test_every_link_points_at_an_existing_output():
    for nid, node in API.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                src, slot = value
                assert src in API, f"{nid}.{name} links to missing node {src}"
                assert slot < len(OBJECT_INFO[API[src]["class_type"]]["output"])


@pytest.mark.parametrize("title", graph.TITLES)
def test_every_title_the_cli_patches_exists_once(title):
    graph.find(API, title)


def test_all_three_upscalers_derive_their_size_from_the_stage_1_node():
    stage1 = graph.find(API, graph.STAGE1)
    for nid, node in API.items():
        if node["class_type"] == "ImageUpscaleWithModel":
            quarter = node["inputs"]["image"][0]
            assert API[quarter]["class_type"] == "ImageScaleBy"
            assert API[quarter]["inputs"]["scale_by"] == 0.25
            assert API[quarter]["inputs"]["image"][0] == stage1, nid
        if node["class_type"] in {"SeedVR2Preprocess"}:
            assert node["inputs"]["resized_images"][0] == stage1


def test_the_ui_workflow_has_the_same_nodes_as_the_api_graph():
    ui = {str(n["id"]): n["type"] for n in UI["nodes"] if n["type"] not in {"MarkdownNote", "Note"}}
    assert ui == {nid: n["class_type"] for nid, n in API.items()}


def test_every_model_loader_carries_a_download_link_and_hash():
    loaders = [n for n in UI["nodes"] if n["type"] in LOADERS]
    assert len(loaders) == 4
    for node in loaders:
        (model,) = node["properties"]["models"]
        assert model["url"].startswith("https://")
        assert model["name"] == node["widgets_values"][0]
        assert len(model["hash"]) == 64
