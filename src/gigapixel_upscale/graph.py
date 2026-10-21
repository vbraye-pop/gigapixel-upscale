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
NOMOS_LOW = "Nomos2, low frequencies"
DETAIL_LOW = "Detail mix, low frequencies"
TITLES = (SOURCE, STAGE1, CAMERA, DETAIL, MASTER, POSTER_SCALE, POSTER_PNG, NOMOS_LOW, DETAIL_LOW)

SEEDVR2 = {
    "seedvr2-7b": "seedvr2_7b_int8_convrot.safetensors",
    "seedvr2-7b-sharp": "seedvr2_7b_sharp_int8_convrot.safetensors",
    "seedvr2-3b": "seedvr2_3b_fp16.safetensors",
}


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


def find_class(graph: dict, class_type: str) -> str:
    hits = [nid for nid, node in graph.items() if node["class_type"] == class_type]
    if len(hits) != 1:
        raise GraphError(f"class {class_type!r} matched {len(hits)} nodes")
    return hits[0]


def use_detail_image(graph: dict, image: str) -> None:
    """Feed the detail mix from an uploaded image instead of SeedVR2, resized to the stage 1 size."""
    stage1 = find(graph, STAGE1)
    graph["900"] = {"class_type": "LoadImage", "inputs": {"image": image}, "_meta": {"title": "Detail source"}}
    graph["901"] = {
        "class_type": "ImageResizeKJv2",
        "inputs": {
            "image": ["900", 0],
            "width": [stage1, 1],
            "height": [stage1, 2],
            "upscale_method": "lanczos",
            "keep_proportion": "stretch",
            "pad_color": "0, 0, 0",
            "crop_position": "center",
            "divisible_by": 8,
        },
        "_meta": {"title": "Detail source, stage 1 size"},
    }
    graph[find(graph, DETAIL)]["inputs"]["image2"] = ["901", 0]


def prune(graph: dict, keep: str) -> None:
    """Drop every node the kept output does not depend on, so the replaced SeedVR2 branch never loads."""
    needed, todo = set(), [keep]
    while todo:
        nid = todo.pop()
        if nid in needed:
            continue
        needed.add(nid)
        todo += [v[0] for v in graph[nid]["inputs"].values() if isinstance(v, list)]
    for nid in set(graph) - needed:
        del graph[nid]


def prepare(
    template: dict,
    image: str,
    camera: bool = False,
    detail: float = 0.75,
    box: tuple[int, int] = (8192, 8192),
    prefix: str = "gigapixel/master",
    seedvr2: str = "seedvr2-7b",
    sigma: float = 0.75,
    detail_image: str | None = None,
) -> dict:
    """A ready-to-queue copy: the source set, the stage 1 box set, and the poster nodes removed.

    With detail_image, an image already uploaded to the server replaces SeedVR2 as the detail source.
    The poster is made locally from the downloaded master, which is faster and adds the DPI tag.
    """
    graph = copy.deepcopy(template)
    set_input(graph, SOURCE, "image", image)
    set_input(graph, CAMERA, "switch", camera)
    set_input(graph, DETAIL, "blend_factor", detail)
    set_input(graph, NOMOS_LOW, "sigma", sigma)
    set_input(graph, DETAIL_LOW, "sigma", sigma)
    set_input(graph, STAGE1, "width", box[0])
    set_input(graph, STAGE1, "height", box[1])
    set_input(graph, MASTER, "filename_prefix", prefix)
    graph[find_class(graph, "UNETLoader")]["inputs"]["unet_name"] = SEEDVR2[seedvr2]
    for title in (POSTER_PNG, POSTER_SCALE):
        del graph[find(graph, title)]
    if detail_image:
        use_detail_image(graph, detail_image)
        prune(graph, find(graph, MASTER))
    return graph
