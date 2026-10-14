import pytest
import pyvips

from gigapixel_upscale.poster import ICC, make_poster, rgb_profile


@pytest.fixture
def master(tmp_path):
    path = tmp_path / "master.png"
    ramp = pyvips.Image.xyz(64, 48)[0].cast("uchar")
    ramp.bandjoin([ramp, ramp]).copy(interpretation="srgb").pngsave(str(path))
    return path


def tagged_source(tmp_path, profile, bands=3, interpretation="srgb"):
    path = tmp_path / f"source_{profile}.tif"
    pyvips.Image.black(8, 8, bands=bands).copy(interpretation=interpretation).tiffsave(str(path), profile=profile)
    return path


def test_the_poster_lands_on_the_pixel_budget_with_the_master_aspect(master, tmp_path):
    poster = make_poster(master, tmp_path / "p.tif", megapixels=0.0768)
    assert (poster.width, poster.height) == (320, 240)


def test_a_long_side_sets_the_size_instead_of_the_budget(master, tmp_path):
    poster = make_poster(master, tmp_path / "p.tif", long_side=640)
    assert (poster.width, poster.height) == (640, 480)


def test_the_tiff_carries_300_dpi_in_inches(master, tmp_path):
    out = tmp_path / "p.tif"
    poster = make_poster(master, out, long_side=128)
    saved = pyvips.Image.new_from_file(str(out))
    assert poster.dpi == 300
    assert saved.get("resolution-unit") == "in"


def test_the_poster_is_a_classic_lzw_tiff_unless_bigtiff_is_asked(master, tmp_path):
    classic = tmp_path / "classic.tif"
    big = tmp_path / "big.tif"
    make_poster(master, classic, long_side=128)
    make_poster(master, big, long_side=128, bigtiff=True)
    assert classic.read_bytes()[:4] == b"II*\x00"
    assert big.read_bytes()[:4] == b"II+\x00"


def test_an_untagged_source_gets_srgb(master, tmp_path):
    poster = make_poster(master, tmp_path / "p.tif", long_side=128)
    saved = pyvips.Image.new_from_file(str(poster.path))
    assert poster.profile == "sRGB"
    assert saved.get(ICC)[16:20] == b"RGB "


def test_an_rgb_source_profile_is_carried_over_unchanged(master, tmp_path):
    source = tagged_source(tmp_path, "p3")
    poster = make_poster(master, tmp_path / "p.tif", long_side=128, profile_from=source)
    assert poster.profile == "source"
    assert pyvips.Image.new_from_file(str(poster.path)).get(ICC) == rgb_profile(source)


def test_a_cmyk_source_profile_is_ignored_in_favour_of_srgb(master, tmp_path):
    source = tagged_source(tmp_path, "cmyk", bands=4, interpretation="cmyk")
    assert rgb_profile(source) is None
    assert make_poster(master, tmp_path / "p.tif", long_side=128, profile_from=source).profile == "sRGB"
