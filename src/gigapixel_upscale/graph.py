"""Load the bundled API-format workflow and patch it by node title.

Titles rather than node ids, because editing a workflow in ComfyUI renumbers its nodes.
"""

import copy
import json
from importlib import resources

SOURCE = "Source image"
STAGE1 = "Stage 1 size (fit in 8192 box)"
CAMERA = "Camera photo switch (true = RealESRGAN)"
DETAIL = "Detail mix"
MASTER = "Master"
POSTER_SCALE = "Poster scale (see table in the read me)"
POSTER_PNG = "Poster PNG"
TITLES = (SOURCE, STAGE1, CAMERA, DETAIL, MASTER, POSTER_SCALE, POSTER_PNG)


class GraphError(ValueError):
    pass


def load() -> dict:
    return json.loads(resources.files(__package__).joinpath("workflow_api.json").read_text())


def title_of(node: dict) -> str:
    return node.get("_meta", {}).get("title", node["class_type"])


def find(graph: dict, title: str) -> str:
    hits = [nid for nid, node in graph.items() if title_of(node) == title]
    if len(hits) != 1:
        available = ", ".join(sorted(title_of(n) for n in graph.values()))
        raise GraphError(f"title {title!r} matched {len(hits)} nodes, available: {available}")
    return hits[0]


def set_input(graph: dict, title: str, name: str, value) -> None:
    inputs = graph[find(graph, title)]["inputs"]
    if name not in inputs:
        raise GraphError(f"{title!r} has no input {name!r}, has: {', '.join(sorted(inputs))}")
    if isinstance(inputs[name], list):
        raise GraphError(f"{title!r}.{name} is a link to another node, not a value")
    inputs[name] = value


def prepare(
    template: dict,
    image: str,
    camera: bool = False,
    detail: float = 0.75,
    box: tuple[int, int] = (8192, 8192),
    prefix: str = "gigapixel/master",
) -> dict:
    """A ready-to-queue copy: the source set, the stage 1 box set, and the poster nodes removed.

    The poster is made locally from the downloaded master, which is faster and adds the DPI tag.
    """
    graph = copy.deepcopy(template)
    set_input(graph, SOURCE, "image", image)
    set_input(graph, CAMERA, "switch", camera)
    set_input(graph, DETAIL, "blend_factor", detail)
    set_input(graph, STAGE1, "width", box[0])
    set_input(graph, STAGE1, "height", box[1])
    set_input(graph, MASTER, "filename_prefix", prefix)
    for title in (POSTER_PNG, POSTER_SCALE):
        del graph[find(graph, title)]
    return graph
