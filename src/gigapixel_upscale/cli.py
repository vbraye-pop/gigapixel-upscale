"""gigapixel-upscale: run the ComfyUI stage on a remote server, then build the poster TIFF locally."""

import argparse
import os
import sys
from pathlib import Path

import pyvips

from . import apis, graph
from .apis import ApiError
from .comfy import Comfy, ComfyError
from .graph import GraphError
from .poster import make_poster
from .sizes import BOX, POSTER_MEGAPIXELS, stage1_size

# best detail share per source, measured in bench/
BEST_DETAIL = {
    "seedvr2-7b": 0.75,
    "seedvr2-7b-sharp": 0.5,
    "seedvr2-3b": 0.75,
    "topaz-wonder": 0.75,
    "topaz-redefine": 0.25,
}


def progress(status: str, elapsed: float) -> None:
    print(f"\r  {status.replace('_', ' ')}, {elapsed:.0f} s", end="", file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> Path:
    source = args.image
    header = pyvips.Image.new_from_file(str(source))
    width, height, capped = stage1_size(header.width, header.height)
    if capped:
        print(
            f"warning: {header.width} x {header.height} is small, master capped at 4x ({width} x {height}) "
            "so SeedVR2 does not invent material",
            file=sys.stderr,
        )
    box = (width, height) if capped else (BOX, BOX)

    detail = BEST_DETAIL[args.detail_source] if args.detail is None else args.detail
    comfy = Comfy(args.server)
    print(f"uploading {source.name} to {comfy.url}", file=sys.stderr)
    name = comfy.upload(source)
    detail_image = None
    if args.detail_source in apis.MODELS and not args.camera:
        print(f"{args.detail_source}: upscaling {source.name} to {width} x {height}", file=sys.stderr)
        result = apis.topaz(args.detail_source, header.pngsave_buffer(), width, height)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        detail_path = args.out_dir / f"{source.stem}_{args.detail_source}.png"
        detail_path.write_bytes(result)
        detail_image = comfy.upload(detail_path)
    seedvr2 = args.detail_source if args.detail_source in graph.SEEDVR2 else "seedvr2-7b"
    prompt = graph.prepare(
        graph.load(),
        name,
        camera=args.camera,
        detail=detail,
        box=box,
        seedvr2=seedvr2,
        sigma=args.sigma,
        detail_image=detail_image,
    )
    prompt_id = comfy.submit(prompt)
    mode = "camera photo (RealESRGAN)" if args.camera else f"blend ({args.detail_source}, detail {detail})"
    print(f"queued {prompt_id}, {mode}, master about {width} x {height}", file=sys.stderr)
    try:
        outputs = comfy.wait(prompt_id, on_status=progress)
    except KeyboardInterrupt:
        print("\ncancelling", file=sys.stderr)
        comfy.cancel(prompt_id)
        raise
    print(file=sys.stderr)

    (image,) = outputs[graph.find(prompt, graph.MASTER)]["images"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    master = comfy.download(image, args.out_dir / f"{source.stem}_master.png")
    print(f"master: {master}", file=sys.stderr)
    return master


def poster(args: argparse.Namespace, master: Path, profile_from: Path | None) -> None:
    out = args.output or master.with_name(master.stem.removesuffix("_master") + f"_poster_{args.dpi:g}dpi.tif")
    print(f"writing {out}", file=sys.stderr)
    result = make_poster(
        master,
        out,
        megapixels=args.megapixels,
        long_side=args.long_side,
        dpi=args.dpi,
        profile_from=profile_from,
        bigtiff=args.bigtiff,
    )
    print(result.describe())


def add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("image", type=Path, help="source image, at least 2048 px on the long side")
    p.add_argument("--server", default=os.environ.get("COMFY_URL"), help="ComfyUI URL (default: $COMFY_URL)")
    p.add_argument("--camera", action="store_true", help="soft or noisy camera shot: use RealESRGAN, skip the blend")
    p.add_argument(
        "--detail-source",
        choices=list(BEST_DETAIL),
        default="seedvr2-7b",
        help="model for the fine detail (default: seedvr2-7b). topaz-* need $TOPAZ_API_KEY",
    )
    p.add_argument(
        "--detail", type=float, help="share of the detail source, 0 = plain Nomos2 (default: best per source)"
    )
    p.add_argument("--sigma", type=float, default=0.75, help="blur sigma splitting tones from detail (default: 0.75)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("out"), help="where the master goes (default: out)")


def add_poster_options(p: argparse.ArgumentParser) -> None:
    size = p.add_mutually_exclusive_group()
    size.add_argument(
        "--megapixels",
        type=float,
        default=POSTER_MEGAPIXELS,
        help=f"pixel budget (default: {POSTER_MEGAPIXELS:.0f}, 4 x 3 m at 300 DPI)",
    )
    size.add_argument("--long-side", type=int, help="long side in pixels instead of a pixel budget")
    p.add_argument("--dpi", type=float, default=300, help="DPI written in the file, no resampling (default: 300)")
    p.add_argument("--bigtiff", action="store_true", help="write BigTIFF, only needed past 4 GB")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gigapixel-upscale", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="ComfyUI stage only: source to 8192 px master")
    add_run_options(r)

    t = sub.add_parser("poster", help="local stage only: master to poster TIFF")
    t.add_argument("master", type=Path)
    add_poster_options(t)
    t.add_argument("--profile-from", type=Path, help="original source, to carry its colour profile over")
    t.add_argument("-o", "--output", type=Path, help="poster path (default: next to the master)")

    a = sub.add_parser("all", help="both stages: source to poster TIFF")
    add_run_options(a)
    add_poster_options(a)
    a.set_defaults(output=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            run(args)
        elif args.command == "poster":
            poster(args, args.master, args.profile_from)
        else:
            poster(args, run(args), args.image)
    except KeyboardInterrupt:
        sys.exit(130)
    except (ApiError, ComfyError, GraphError, pyvips.Error, OSError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
