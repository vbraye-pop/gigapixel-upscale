"""Stage 2: resize the AI master to the poster pixel budget and write a print-ready TIFF.

Everything streams top to bottom through libvips, so a 1.67 gigapixel poster fits on a laptop.
libvips enlarges with bicubic whatever kernel is asked for; against PIL Lanczos (what the ComfyUI
workflow uses) the difference is under 1/255 on average, invisible in print.
"""

from dataclasses import dataclass
from pathlib import Path

import pyvips

from .sizes import POSTER_MEGAPIXELS, physical_cm, poster_size

ICC = "icc-profile-data"


@dataclass
class Poster:
    path: Path
    width: int
    height: int
    dpi: float
    profile: str

    def describe(self) -> str:
        w, h = physical_cm(self.width, self.dpi), physical_cm(self.height, self.dpi)
        return (
            f"{self.path}: {self.width} x {self.height} px ({self.width * self.height / 1e9:.2f} GP), "
            f"{w:.0f} x {h:.0f} cm at {self.dpi:g} DPI, {self.profile} profile"
        )


def rgb_profile(source: Path) -> bytes | None:
    """The source's embedded ICC profile, only if it describes RGB data like the master."""
    image = pyvips.Image.new_from_file(str(source))
    if not image.get_typeof(ICC):
        return None
    data = image.get(ICC)
    # the colour space signature sits at bytes 16-20 of every ICC header
    return data if data[16:20] == b"RGB " else None


def make_poster(
    master: Path,
    out: Path,
    megapixels: float = POSTER_MEGAPIXELS,
    long_side: int | None = None,
    dpi: float = 300,
    profile_from: Path | None = None,
    bigtiff: bool = False,
) -> Poster:
    image = pyvips.Image.new_from_file(str(master), access="sequential")
    if image.hasalpha():
        image = image.extract_band(0, n=image.bands - 1)
    width, height = poster_size(image.width, image.height, megapixels, long_side)
    image = image.resize(width / image.width, vscale=height / image.height, kernel="cubic")

    # ComfyUI drops colour profiles, so the pixels are still in the source's space: assign, never convert
    icc = rgb_profile(profile_from) if profile_from else None
    options = {}
    if icc:
        image = image.copy()
        image.set_type(pyvips.GValue.blob_type, ICC, icc)
    else:
        options["profile"] = "srgb"

    image.tiffsave(
        str(out),
        compression="lzw",
        predictor="horizontal",
        tile=False,
        bigtiff=bigtiff,
        xres=dpi / 25.4,
        yres=dpi / 25.4,
        resunit="inch",
        **options,
    )
    return verify(out, "source" if icc else "sRGB")


def verify(path: Path, profile: str) -> Poster:
    saved = pyvips.Image.new_from_file(str(path))
    if not saved.get_typeof(ICC):
        raise RuntimeError(f"{path} was written without a colour profile")
    return Poster(path, saved.width, saved.height, round(saved.get("xres") * 25.4, 2), profile)
