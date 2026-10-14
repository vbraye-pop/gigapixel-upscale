#!/usr/bin/env bash
# Install what the workflow needs into a ComfyUI folder: ComfyUI-KJNodes (pinned) and the four models.
#
#   COMFYUI_DIR=/path/to/ComfyUI scripts/install.sh
#
# COMFYUI_PYTHON defaults to the python next to ComfyUI (.venv or venv), else python3 on the PATH.
# Safe to run again: existing files are checked, not downloaded twice.
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:?set COMFYUI_DIR to your ComfyUI folder}"
KJNODES_COMMIT="def8a9c22fbfac4568bef93e87c7e1812344d569"  # ComfyUI-KJNodes 1.5.0
TESTED_COMFYUI="0.30.0"

if [[ -z "${COMFYUI_PYTHON:-}" ]]; then
    for candidate in "$COMFYUI_DIR"/.venv*/bin/python "$COMFYUI_DIR/venv/bin/python" "$(command -v python3)"; do
        if [[ -x "$candidate" ]]; then COMFYUI_PYTHON="$candidate"; break; fi
    done
fi

# environments made by uv ship without pip
pip_install() {
    if "$COMFYUI_PYTHON" -m pip --version >/dev/null 2>&1; then
        "$COMFYUI_PYTHON" -m pip install --quiet "$@"
    elif command -v uv >/dev/null; then
        uv pip install --quiet --python "$COMFYUI_PYTHON" "$@"
    else
        echo "no pip in $COMFYUI_PYTHON and no uv on the PATH, install the packages in $*" >&2
        exit 1
    fi
}

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# folder, file name, url, sha256
MODELS=(
    "upscale_models 4xNomos2_hq_dat2.safetensors https://huggingface.co/Phips/4xNomos2_hq_dat2/resolve/main/4xNomos2_hq_dat2.safetensors 278c78ba9fd333ace4b02b3aeaa0254428baea20a63d3e995e2e4f8a60173f6f"
    "upscale_models RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
    "diffusion_models seedvr2_7b_int8_convrot.safetensors https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/diffusion_models/seedvr2_7b_int8_convrot.safetensors 5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6"
    "vae seedvr2_ema_vae_fp16.safetensors https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/vae/seedvr2_ema_vae_fp16.safetensors 20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"
)

echo "== ComfyUI-KJNodes"
existing=$(find "$COMFYUI_DIR/custom_nodes" -maxdepth 1 -iname "comfyui-kjnodes" | head -n1)
if [[ -n "$existing" ]]; then
    echo "already installed at $existing (needs version 1.1.4 or newer)"
else
    existing="$COMFYUI_DIR/custom_nodes/ComfyUI-KJNodes"
    git clone --quiet https://github.com/kijai/ComfyUI-KJNodes "$existing"
    git -C "$existing" checkout --quiet "$KJNODES_COMMIT"
    echo "cloned at $KJNODES_COMMIT"
fi
echo "installing its python packages into $COMFYUI_PYTHON"
pip_install -r "$existing/requirements.txt"

for entry in "${MODELS[@]}"; do
    read -r folder name url hash <<<"$entry"
    dest="$COMFYUI_DIR/models/$folder/$name"
    echo "== $folder/$name"
    mkdir -p "$(dirname "$dest")"
    if [[ ! -f "$dest" ]]; then
        curl --fail --location --continue-at - --progress-bar --output "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
    if [[ "$(sha256 "$dest")" != "$hash" ]]; then
        echo "checksum mismatch for $dest, delete it and run again" >&2
        exit 1
    fi
    echo "ok"
done

echo
echo "Done. Restart ComfyUI. Tested on ComfyUI $TESTED_COMFYUI, needs 0.28.0 or newer for the SeedVR2 nodes."
