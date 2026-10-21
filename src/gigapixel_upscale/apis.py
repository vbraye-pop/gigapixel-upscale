"""Topaz as an alternative detail source. The key comes from $TOPAZ_API_KEY."""

import os
import time

import requests

TOPAZ = "https://api.topazlabs.com/image/v1"
# same settings as the benchmark
MODELS = {
    "topaz-wonder": ("Wonder 3.5", {}),
    "topaz-redefine": ("Redefine", {"creativity": 1, "texture": 1, "autoprompt": "false", "prompt": ""}),
}


class ApiError(RuntimeError):
    pass


def topaz(source: str, png: bytes, width: int, height: int, timeout: float = 1800) -> bytes:
    """Upscale png to width x height with a Topaz generative model, return the result as image bytes."""
    key = os.environ.get("TOPAZ_API_KEY")
    if not key:
        raise ApiError(f"{source} needs a Topaz API key in $TOPAZ_API_KEY")
    model, params = MODELS[source]
    headers = {"X-API-KEY": key}
    size = {"output_width": width, "output_height": height, "outputWidth": width, "outputHeight": height}
    fields = {"model": model, "output_format": "png", **size, **params}
    files = {"image": ("source.png", png, "image/png")}
    r = requests.post(f"{TOPAZ}/enhance-gen/async", headers=headers, data=fields, files=files, timeout=300)
    if not r.ok:
        raise ApiError(f"Topaz refused the job: {r.status_code} {r.text[:300]}")
    pid = r.json()["process_id"]
    start = time.monotonic()
    while True:
        status = requests.get(f"{TOPAZ}/status/{pid}", headers=headers, timeout=60).json().get("status")
        if status == "Completed":
            break
        if status in ("Failed", "Cancelled"):
            raise ApiError(f"Topaz job {pid} ended {status}")
        if time.monotonic() - start > timeout:
            raise ApiError(f"Topaz job {pid} still {status} after {timeout:.0f} s")
        time.sleep(3)
    d = requests.get(f"{TOPAZ}/download/{pid}", headers=headers, timeout=60).json()
    url = d.get("url") or d.get("download_url") or d.get("downloadUrl")
    if not url:
        raise ApiError(f"no download url for Topaz job {pid}: {d}")
    return requests.get(url, timeout=600).content
