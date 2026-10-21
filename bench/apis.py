"""Closed upscale services, called with the same low-res inputs as the ComfyUI candidates.

Each function takes PNG bytes and the target size, and returns PNG or JPEG bytes. Times include queueing on the
provider's side, so they are not comparable with the GPU timings.
"""

import base64
import os
import time

import requests

TOPAZ = "https://api.topazlabs.com/image/v1"
MAGNIFIC = "https://api.magnific.com/v1/ai/image-upscaler"


def _poll(get, done, failed, every=3, limit=1800):
    start = time.time()
    while time.time() - start < limit:
        r = get()
        if done(r):
            return r
        if failed(r):
            raise RuntimeError(str(r)[:1000])
        time.sleep(every)
    raise TimeoutError("provider did not finish in time")


def topaz(model, **params):
    def run(png, w, h):
        headers = {"X-API-KEY": os.environ["TOPAZ_API_KEY"]}
        fields = {
            "model": model,
            "output_format": "png",
            "output_width": w,
            "output_height": h,
            "outputWidth": w,
            "outputHeight": h,
            **params,
        }
        r = requests.post(
            f"{TOPAZ}/enhance-gen/async",
            headers=headers,
            data=fields,
            files={"image": ("in.png", png, "image/png")},
            timeout=120,
        )
        r.raise_for_status()
        pid = r.json()["process_id"]
        _poll(
            lambda: requests.get(f"{TOPAZ}/status/{pid}", headers=headers, timeout=60).json(),
            lambda s: s.get("status") == "Completed",
            lambda s: s.get("status") in ("Failed", "Cancelled"),
        )
        d = requests.get(f"{TOPAZ}/download/{pid}", headers=headers, timeout=60).json()
        url = d.get("url") or d.get("download_url") or d.get("downloadUrl")
        if not url:
            raise RuntimeError(f"no download url in {d}")
        return requests.get(url, timeout=300).content

    return run


def magnific(endpoint, **params):
    def run(png, w, h):
        headers = {"x-magnific-api-key": os.environ["MAGNIFIC_API_KEY"], "Content-Type": "application/json"}
        url = f"{MAGNIFIC}{endpoint}"
        body = {"image": base64.b64encode(png).decode(), "filter_nsfw": False, **params}
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        tid = r.json()["data"]["task_id"]
        s = _poll(
            lambda: requests.get(f"{url}/{tid}", headers=headers, timeout=60).json()["data"],
            lambda d: d.get("status") == "COMPLETED",
            lambda d: d.get("status") == "FAILED",
            every=5,
        )
        return requests.get(s["generated"][0], timeout=300).content

    return run


def twice(run):
    """Precision v2 returns 2x whatever scale_factor asks for, so 4x is two 2x passes."""

    def chained(png, w, h):
        return run(run(png, w // 2, h // 2), w, h)

    return chained


METHODS = {
    "topaz_wonder35": topaz("Wonder 3.5"),
    "topaz_redefine": topaz("Redefine", creativity=1, texture=1, autoprompt="false", prompt=""),
    "magnific": magnific(
        "",
        scale_factor="4x",
        optimized_for="films_n_photography",
        creativity=0,
        hdr=0,
        resemblance=5,
        fractality=0,
        engine="automatic",
    ),
    "magnific_precision_2x2": twice(
        magnific("-precision-v2", scale_factor=2, flavor="photo", sharpen=7, smart_grain=7, ultra_detail=30)
    ),
}
