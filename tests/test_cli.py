import pytest
import pyvips

from gigapixel_upscale import cli, graph


def write_image(path, width, height):
    pyvips.Image.black(width, height, bands=3).copy(interpretation="srgb").pngsave(str(path))
    return path


class FakeComfy:
    """Stands in for the server: records the queued graph and returns a small master."""

    def __init__(self, url):
        self.url = url
        self.prompt = None

    def upload(self, path):
        return f"up_{path.name}"

    def submit(self, prompt):
        FakeComfy.last = self
        self.prompt = prompt
        return "pid"

    def wait(self, prompt_id, on_status=None):
        return {
            graph.find(self.prompt, graph.MASTER): {
                "images": [{"filename": "m.png", "subfolder": "", "type": "output"}]
            }
        }

    def cancel(self, prompt_id):
        pass

    def download(self, ref, dest):
        return write_image(dest, 64, 48)


@pytest.fixture
def fake_server(monkeypatch):
    monkeypatch.setattr(cli, "Comfy", FakeComfy)


def queued(title, name):
    p = FakeComfy.last.prompt
    return p[graph.find(p, title)]["inputs"][name]


def test_run_queues_the_uploaded_source_and_downloads_the_master(fake_server, tmp_path):
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    cli.main(["run", str(source), "--server", "http://x", "-o", str(tmp_path / "out")])
    assert (tmp_path / "out" / "cat_master.png").exists()
    assert queued(graph.SOURCE, "image") == "up_cat.png"
    assert (queued(graph.STAGE1, "width"), queued(graph.STAGE1, "height")) == (8192, 8192)
    assert queued(graph.CAMERA, "switch") is False


def test_a_small_source_gets_a_box_capped_at_4x(fake_server, tmp_path):
    source = write_image(tmp_path / "small.png", 1000, 750)
    cli.main(["run", str(source), "--server", "http://x", "--camera", "-o", str(tmp_path)])
    assert (queued(graph.STAGE1, "width"), queued(graph.STAGE1, "height")) == (4000, 3000)
    assert queued(graph.CAMERA, "switch") is True


def test_all_ends_with_a_poster_next_to_the_master(fake_server, tmp_path, capsys):
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    cli.main(["all", str(source), "--server", "http://x", "-o", str(tmp_path), "--long-side", "128", "--dpi", "150"])
    assert (tmp_path / "cat_poster_150dpi.tif").exists()
    assert "128 x 96 px" in capsys.readouterr().out


def test_poster_alone_works_on_a_local_master(tmp_path, capsys):
    master = write_image(tmp_path / "m_master.png", 64, 48)
    cli.main(["poster", str(master), "--long-side", "256", "-o", str(tmp_path / "p.tif")])
    assert "256 x 192 px" in capsys.readouterr().out


def test_a_missing_server_is_a_clean_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("COMFY_URL", raising=False)
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run", str(source)])
    assert exit_.value.code == 1
    assert "COMFY_URL" in capsys.readouterr().err


def test_each_detail_source_gets_its_benchmark_share(fake_server, tmp_path):
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    cli.main(["run", str(source), "--server", "http://x", "--detail-source", "seedvr2-7b-sharp", "-o", str(tmp_path)])
    assert queued(graph.DETAIL, "blend_factor") == 0.5


def test_a_topaz_source_without_a_key_is_a_clean_error(fake_server, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TOPAZ_API_KEY", raising=False)
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    with pytest.raises(SystemExit):
        cli.main(["run", str(source), "--server", "http://x", "--detail-source", "topaz-wonder", "-o", str(tmp_path)])
    assert "TOPAZ_API_KEY" in capsys.readouterr().err


def test_a_topaz_result_is_uploaded_and_feeds_the_blend(fake_server, tmp_path, monkeypatch):
    sizes = []

    def fake_topaz(name, png, w, h):
        sizes.append((w, h))
        return pyvips.Image.black(w, h, bands=3).pngsave_buffer()

    monkeypatch.setattr(cli.apis, "topaz", fake_topaz)
    source = write_image(tmp_path / "cat.png", 2048, 1536)
    cli.main(["run", str(source), "--server", "http://x", "--detail-source", "topaz-wonder", "-o", str(tmp_path)])
    assert sizes == [(8192, 6144)]
    assert queued("Detail source", "image") == "up_cat_topaz-wonder.png"
    assert queued(graph.DETAIL, "blend_factor") == 0.75
