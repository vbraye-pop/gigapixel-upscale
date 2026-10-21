"""Candidate detail sources for the blend, run headless in ComfyUI on Modal.

modal run bench/app.py::download                                    fetch models into the models volume
modal run bench/app.py::dump_info --out bench/object_info.json      node definitions, to check graphs offline
modal run bench/app.py::main --methods vosr2,iris --ids-file bench/screen30.txt --warm krea_jewelry_ring
modal run bench/app.py::api_main --methods magnific --ids-file bench/screen30.txt
modal run bench/app.py::scores --methods vosr2,iris --ids-file bench/screen30.txt
"""

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import modal

COMFY_TAG = "v0.42.1"
NODES = {
    "ComfyUI-KJNodes": ("https://github.com/kijai/ComfyUI-KJNodes", "def8a9c22fbfac4568bef93e87c7e1812344d569"),
    "ComfyUI-VOSR2": ("https://github.com/ylchen333/ComfyUI-VOSR2", None),
    "Iris-3B-Comfy-Nodes": ("https://github.com/bani4kaskashka/Iris-3B-Comfy-Nodes", None),
}
COMFY = "/root/ComfyUI"
MODELS = f"{COMFY}/models"
DATA = "/data"
GPU = "H100"


def clone_nodes():
    cmds = []
    for name, (url, ref) in NODES.items():
        d = f"{COMFY}/custom_nodes/{name}"
        cmds.append(f"git clone {url} {d}")
        if ref:
            cmds.append(f"git -C {d} checkout {ref}")
        cmds.append(f"if [ -f {d}/requirements.txt ]; then pip install -r {d}/requirements.txt; fi")
    return cmds


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1", index_url="https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        f"git clone --depth 1 --branch {COMFY_TAG} https://github.com/comfyanonymous/ComfyUI {COMFY}",
        f"pip install -r {COMFY}/requirements.txt",
        f"rm -rf {COMFY}/models",
    )
    .run_commands(*clone_nodes())
    .pip_install(
        "huggingface_hub[hf_transfer]", "requests", "pyiqa", "scipy", "pillow", "numpy<2.3", "opencv-python-headless"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("graphs", "apis", "score")
)

app = modal.App("gigapixel-bench", image=image)
models_vol = modal.Volume.from_name("gigapixel-bench-models", create_if_missing=True)
data_vol = modal.Volume.from_name("gigapixel-bench-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("gigapixel-hf", required_keys=["HF_TOKEN"])
api_secret = modal.Secret.from_name("gigapixel-apis", required_keys=["TOPAZ_API_KEY", "MAGNIFIC_API_KEY"])
VOLS = {MODELS: models_vol, DATA: data_vol}

# folder under models/, file name, HF repo (or a direct URL), path in repo
FILES = [
    (
        "upscale_models",
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "",
    ),
    ("upscale_models", "4xNomos2_hq_dat2.safetensors", "Phips/4xNomos2_hq_dat2", "4xNomos2_hq_dat2.safetensors"),
    (
        "diffusion_models",
        "seedvr2_7b_int8_convrot.safetensors",
        "Comfy-Org/SeedVR2",
        "diffusion_models/seedvr2_7b_int8_convrot.safetensors",
    ),
    (
        "diffusion_models",
        "seedvr2_7b_sharp_int8_convrot.safetensors",
        "Comfy-Org/SeedVR2",
        "diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors",
    ),
    (
        "diffusion_models",
        "seedvr2_3b_fp16.safetensors",
        "Comfy-Org/SeedVR2",
        "diffusion_models/seedvr2_3b_fp16.safetensors",
    ),
    ("vae", "seedvr2_ema_vae_fp16.safetensors", "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors"),
    (
        "diffusion_models",
        "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
        "Comfy-Org/PixelDiT",
        "diffusion_models/pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
    ),
    (
        "diffusion_models",
        "pid_flux2_512_to_2048_4step_bf16.safetensors",
        "Comfy-Org/PixelDiT",
        "diffusion_models/pid_flux2_512_to_2048_4step_bf16.safetensors",
    ),
    (
        "diffusion_models",
        "flux-2-klein-4b.safetensors",
        "Comfy-Org/flux2-klein-4B",
        "split_files/diffusion_models/flux-2-klein-4b.safetensors",
    ),
    (
        "text_encoders",
        "qwen_3_4b.safetensors",
        "Comfy-Org/flux2-klein-4B",
        "split_files/text_encoders/qwen_3_4b.safetensors",
    ),
    ("vae", "flux2-vae.safetensors", "Comfy-Org/flux2-klein-4B", "split_files/vae/flux2-vae.safetensors"),
    (
        "text_encoders",
        "qwen_3_8b_fp8mixed.safetensors",
        "Comfy-Org/flux2-klein-9B",
        "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
    ),
    (
        "diffusion_models",
        "flux-2-klein-9b-fp8.safetensors",
        "black-forest-labs/FLUX.2-klein-9b-fp8",
        "flux-2-klein-9b-fp8.safetensors",
    ),
    (
        "diffusion_models",
        "qwen_image_2.1_bf16.safetensors",
        "Comfy-Org/Qwen-Image-2.1",
        "diffusion_models/qwen_image_2.1_bf16.safetensors",
    ),
    (
        "text_encoders",
        "qwen3vl_8b_bf16.safetensors",
        "Comfy-Org/Qwen-Image-2.1",
        "text_encoders/qwen3vl_8b_bf16.safetensors",
    ),
    (
        "vae",
        "qwen_image_2.1_vae_bf16.safetensors",
        "Comfy-Org/Qwen-Image-2.1",
        "vae/qwen_image_2.1_vae_bf16.safetensors",
    ),
    (
        "diffusion_models",
        "flux3-image-klein-bf16.safetensors",
        "black-forest-labs/flux-3-image-klein-comfy",
        "diffusion_models/flux3-image-klein-bf16.safetensors",
    ),
    (
        "text_encoders",
        "qwen3vl_4b_bf16.safetensors",
        "black-forest-labs/flux-3-image-klein-comfy",
        "text_encoders/qwen3vl_4b_bf16.safetensors",
    ),
    ("model_patches", "SUPIR-v0Q_fp16.safetensors", "Kijai/SUPIR_pruned", "SUPIR-v0Q_fp16.safetensors"),
    (
        "checkpoints",
        "juggernautXL_v9Rdphoto2Lightning.safetensors",
        "AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S",
        "juggernautXL_v9Rdphoto2Lightning.safetensors",
    ),
    (
        "text_encoders",
        "gemma_2_2b_it_elm_fp8_scaled.safetensors",
        "Comfy-Org/PixelDiT",
        "text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors",
    ),
    ("diffusion_models", "iris-3b-upscaler.safetensors", "speridlabs/iris-3b", "upscaler/model.safetensors"),
]


@app.function(volumes=VOLS, secrets=[hf_secret], timeout=4 * 3600, cpu=8, memory=32768)
def download(only: str = ""):
    from huggingface_hub import hf_hub_download, snapshot_download

    for folder, name, repo, path in FILES:
        if only and only not in name:
            continue
        dest = f"{MODELS}/{folder}/{name}"
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            if repo.startswith("https://"):
                urllib.request.urlretrieve(repo, dest)
            else:
                os.replace(hf_hub_download(repo, path, local_dir=f"{MODELS}/.hf/{repo}"), dest)
        except Exception as e:  # noqa: BLE001  a gated or missing file must not stop the other downloads
            print("FAILED", repo, path, str(e)[:200], flush=True)
            continue
        print("ok", dest, flush=True)
        models_vol.commit()
    if not only or "iris" in only:
        snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir=f"{MODELS}/text_encoders/Qwen3-VL-4B-Instruct")
        models_vol.commit()


def load_times(path):
    return json.loads(Path(path).read_text()) if os.path.exists(path) else {}


class Comfy:
    def __init__(self, extra_args=()):
        self.proc = subprocess.Popen(
            [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--output-directory",
                f"{DATA}/comfy_out",
                "--input-directory",
                f"{DATA}/comfy_in",
                *extra_args,
            ],
            cwd=COMFY,
        )
        self.url = "http://127.0.0.1:8188"
        for _ in range(600):
            try:
                urllib.request.urlopen(self.url + "/system_stats", timeout=2)
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError("ComfyUI did not start")

    def get(self, path):
        return json.loads(urllib.request.urlopen(self.url + path, timeout=60).read(), strict=False)

    def run(self, graph):
        """Queue a graph, return (seconds from execution start to success, outputs, peak VRAM GB)."""
        body = json.dumps({"prompt": graph}).encode()
        req = urllib.request.Request(self.url + "/prompt", data=body, headers={"Content-Type": "application/json"})
        try:
            pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:2000]) from None
        peak = 0.0
        while True:
            h = self.get(f"/history/{pid}")
            if h:
                break
            peak = max(peak, used_vram_gb())
            time.sleep(0.5)
        e = h[pid]
        ts = {m[0]: m[1].get("timestamp") for m in e["status"]["messages"]}
        if "execution_success" not in ts:
            raise RuntimeError(json.dumps(e["status"])[:3000])
        return (ts["execution_success"] - ts["execution_start"]) / 1000, e["outputs"], round(peak, 1)

    def close(self):
        self.proc.terminate()


def used_vram_gb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
    ).stdout
    return float(out.split()[0]) / 1024 if out.strip() else 0.0


@app.function(volumes=VOLS, timeout=1800, cpu=4, memory=32768)
def object_info(packs: str = ""):
    c = Comfy(["--cpu"])
    info = c.get("/object_info")
    c.close()
    return {
        k: v
        for k, v in info.items()
        if not packs
        or any(p.lower() in (v.get("python_module") or "").lower() or p.lower() in k.lower() for p in packs.split(","))
    }


@app.function(volumes=VOLS, gpu=GPU, timeout=8 * 3600, memory=131072, cpu=8, secrets=[hf_secret])
def bench(method: str, ids: list[str], warm_id: str = "", run: str = "screen"):
    """Upscale ids with one method, saving outputs to /data/<run>/out and timings to /data/<run>/times."""
    import graphs
    from PIL import Image

    data_vol.reload()
    os.makedirs(f"{DATA}/comfy_in", exist_ok=True)
    os.makedirs(f"{DATA}/{run}/out", exist_ok=True)
    times_path = f"{DATA}/{run}/times/{method}.json"
    os.makedirs(os.path.dirname(times_path), exist_ok=True)
    times = load_times(times_path)
    c = Comfy()
    stamp = int(time.time())

    def one(iid, tag):
        lr = f"{DATA}/set/lr4/{iid}.png"
        w, h = Image.open(f"{DATA}/set/gt/{iid}.png").size
        name = f"{method}_{iid}_{tag}_{stamp}.png"
        Image.open(lr).save(f"{DATA}/comfy_in/{name}")
        g = graphs.build(method, name, w, h, prefix=f"{method}/{iid}_{tag}")
        dt, outs, peak = c.run(g)
        img = next(iter(outs.values()))["images"][0]
        src = f"{DATA}/comfy_out/{img['subfolder']}/{img['filename']}"
        return dt, peak, src, (w, h)

    if warm_id:
        one(warm_id, "warm")
    for iid in ids:
        if "seconds" in times.get(iid, {}):
            continue
        try:
            dt, peak, src, size = one(iid, "run")
        except Exception as e:  # noqa: BLE001  one failed image must not stop the run
            print("FAILED", method, iid, str(e)[:1500], flush=True)
            times[iid] = {"error": str(e)[:500]}
            continue
        out = Image.open(src).convert("RGB")
        if out.size != size:
            print("size mismatch", method, iid, out.size, size, flush=True)
            out = out.resize(size, Image.LANCZOS)
        out.save(f"{DATA}/{run}/out/{iid}__x4_{method}.png")
        times[iid] = {"seconds": round(dt, 2), "peak_vram_gb": peak}
        Path(times_path).write_text(json.dumps(times, indent=1))
        data_vol.commit()
        print(method, iid, round(dt, 1), "s", peak, "GB", flush=True)
    c.close()
    data_vol.commit()
    return times


@app.function(volumes={DATA: data_vol}, secrets=[api_secret], timeout=6 * 3600, cpu=2, memory=8192)
def api_bench(method: str, ids: list[str], run: str = "screen"):
    import io

    import apis
    from PIL import Image

    data_vol.reload()
    os.makedirs(f"{DATA}/{run}/out", exist_ok=True)
    times_path = f"{DATA}/{run}/times/{method}.json"
    os.makedirs(os.path.dirname(times_path), exist_ok=True)
    times = load_times(times_path)
    for iid in ids:
        if "seconds" in times.get(iid, {}):
            continue
        size = Image.open(f"{DATA}/set/gt/{iid}.png").size
        start = time.time()
        try:
            raw = apis.METHODS[method](Path(f"{DATA}/set/lr4/{iid}.png").read_bytes(), *size)
        except Exception as e:  # noqa: BLE001  one failed image must not stop the run
            print("FAILED", method, iid, str(e)[:1000], flush=True)
            times[iid] = {"error": str(e)[:500]}
            continue
        out = Image.open(io.BytesIO(raw)).convert("RGB")
        if out.size != size:
            print("size mismatch", method, iid, out.size, size, flush=True)
            out = out.resize(size, Image.LANCZOS)
        out.save(f"{DATA}/{run}/out/{iid}__x4_{method}.png")
        times[iid] = {"seconds": round(time.time() - start, 2)}
        Path(times_path).write_text(json.dumps(times, indent=1))
        data_vol.commit()
        print(method, iid, times[iid]["seconds"], "s", flush=True)
    return times


@app.function(volumes={DATA: data_vol}, gpu="L40S", timeout=6 * 3600, memory=65536, cpu=8)
def score_run(methods: list[str], ids: list[str], run: str = "screen", name: str = "scores"):
    import score

    data_vol.reload()
    return score.score(f"{DATA}/set", f"{DATA}/{run}/out", methods, ids, f"{DATA}/{run}/{name}.csv", data_vol.commit)


@app.local_entrypoint()
def api_main(methods: str, ids_file: str, run: str = "screen"):
    ids = Path(ids_file).read_text().split()
    names = methods.split(",")
    for name, times in zip(
        names, api_bench.starmap([(m, ids, run) for m in names], return_exceptions=True), strict=True
    ):
        print(name, json.dumps(times, indent=1) if isinstance(times, dict) else repr(times)[:2000])


@app.local_entrypoint()
def scores(methods: str, ids_file: str, run: str = "screen", out: str = "", name: str = "scores"):
    ids = Path(ids_file).read_text().split()
    csv_text = score_run.remote(methods.split(","), ids, run, name)
    path = out or os.path.join(os.path.dirname(__file__), "results", f"{run}_scores.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_text(csv_text)
    print("wrote", path)


@app.local_entrypoint()
def dump_info(out: str = "object_info.json"):
    Path(out).write_text(json.dumps(object_info.remote(), indent=1))


@app.local_entrypoint()
def main(methods: str, ids_file: str, warm: str = "", run: str = "screen"):
    """Run several methods in parallel containers, one per method."""
    ids = Path(ids_file).read_text().split()
    names = methods.split(",")
    for name, times in zip(
        names, bench.starmap([(m, ids, warm, run) for m in names], return_exceptions=True), strict=True
    ):
        print(name, json.dumps(times, indent=1) if isinstance(times, dict) else repr(times)[:2000])
