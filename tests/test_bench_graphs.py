"""The benchmark graphs against the node dump of their container (ComfyUI 0.42.1 plus the bench node packs)."""

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).parent.parent / "bench"
OBJECT_INFO = json.loads((BENCH / "object_info.json").read_text())
_spec = importlib.util.spec_from_file_location("bench_graphs", BENCH / "graphs.py")
graphs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graphs)


@pytest.mark.parametrize("method", sorted(graphs.METHODS))
def test_every_node_and_link_matches_the_node_dump(method):
    g = graphs.build(method, "in.png", 2048, 1536, prefix="out")
    for nid, node in g.items():
        spec = OBJECT_INFO[node["class_type"]]["input"]
        required, optional = spec.get("required", {}), spec.get("optional", {})
        names = {name.split(".")[0] for name in node["inputs"]}
        assert names <= set(required) | set(optional), f"{method} {nid} {node['class_type']}"
        assert set(required) <= names, f"{method} {nid} {node['class_type']}"
        for value in node["inputs"].values():
            if isinstance(value, list):
                src, slot = value
                assert slot < len(OBJECT_INFO[g[src]["class_type"]]["output"])
