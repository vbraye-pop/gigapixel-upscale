import pytest

from gigapixel_upscale.sizes import POSTER_MEGAPIXELS, physical_cm, poster_size, stage1_size


def test_a_4_3_source_fills_the_box_on_its_long_side():
    assert stage1_size(2048, 1536) == (8192, 6144, False)


def test_a_square_source_fills_the_whole_box():
    assert stage1_size(2048, 2048) == (8192, 8192, False)


def test_a_large_source_is_scaled_down_into_the_box():
    assert stage1_size(16000, 9000) == (8192, 4608, False)


def test_a_small_source_is_capped_at_4x_instead_of_filling_the_box():
    assert stage1_size(1024, 768) == (4096, 3072, True)


@pytest.mark.parametrize("w, h", [(1632, 1224), (3000, 2000), (1999, 1333), (4096, 2304)])
def test_stage1_sizes_are_divisible_by_8_and_keep_the_aspect(w, h):
    sw, sh, _ = stage1_size(w, h)
    assert sw % 8 == 0 and sh % 8 == 0
    assert abs(sw / sh - w / h) < 0.01


def test_the_default_poster_of_a_4_3_master_is_the_4x3m_300dpi_size():
    w, h = poster_size(8192, 6144)
    assert abs(w - 47244) <= 1 and abs(h - 35433) <= 1


@pytest.mark.parametrize("w, h", [(8192, 8192), (8192, 6144), (8192, 4608), (6144, 8192)])
def test_posters_land_on_the_pixel_budget_whatever_the_aspect(w, h):
    pw, ph = poster_size(w, h)
    assert abs(pw * ph / 1e6 - POSTER_MEGAPIXELS) / POSTER_MEGAPIXELS < 0.001
    assert abs(pw / ph - w / h) < 0.001


def test_a_long_side_overrides_the_pixel_budget():
    assert poster_size(8192, 4096, long_side=40000) == (40000, 20000)


def test_physical_size_converts_pixels_at_a_dpi_to_centimetres():
    assert physical_cm(47244, 300) == pytest.approx(400, abs=0.01)
