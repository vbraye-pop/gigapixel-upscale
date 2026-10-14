import pytest

from gigapixel_upscale import graph
from gigapixel_upscale.graph import GraphError, find, prepare, set_input


@pytest.fixture
def template():
    return graph.load()


def test_prepare_sets_the_source_switch_detail_and_box(template):
    g = prepare(template, "input/cat.png", camera=True, detail=0.5, box=(4096, 3072))
    assert g[find(g, graph.SOURCE)]["inputs"]["image"] == "input/cat.png"
    assert g[find(g, graph.CAMERA)]["inputs"]["switch"] is True
    assert g[find(g, graph.DETAIL)]["inputs"]["blend_factor"] == 0.5
    stage1 = g[find(g, graph.STAGE1)]["inputs"]
    assert (stage1["width"], stage1["height"]) == (4096, 3072)


def test_prepare_drops_the_poster_nodes_and_keeps_the_master(template):
    g = prepare(template, "x.png")
    titles = {graph.title_of(n) for n in g.values()}
    assert graph.MASTER in titles
    assert graph.POSTER_PNG not in titles and graph.POSTER_SCALE not in titles


def test_prepare_never_touches_the_template(template):
    before = graph.load()
    prepare(template, "x.png", camera=True)
    assert template == before


def test_an_unknown_title_lists_what_is_available(template):
    with pytest.raises(GraphError, match="available: .*Master"):
        find(template, "Nope")


def test_a_duplicated_title_is_refused(template):
    template["999"] = {"class_type": "SaveImage", "_meta": {"title": graph.MASTER}, "inputs": {}}
    with pytest.raises(GraphError, match="matched 2 nodes"):
        find(template, graph.MASTER)


def test_links_cannot_be_overwritten(template):
    with pytest.raises(GraphError, match="is a link"):
        set_input(template, graph.MASTER, "images", "oops")


def test_an_unknown_input_is_refused(template):
    with pytest.raises(GraphError, match="has no input"):
        set_input(template, graph.MASTER, "nope", 1)
