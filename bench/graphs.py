"""ComfyUI API graphs for each candidate: load the low-res image, return an upscale at the ground truth size.

Settings follow the official Comfy templates for each model. Prompted models all get PROMPT, never a per-image caption.
"""

PROMPT = (
    "Upscale this photo to high resolution. Keep the composition, framing, colors and every object exactly as they "
    "are. Add fine, natural texture and sharp detail."
)
SUPIR_POSITIVE = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - "
    "realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore "
    "detailing, hyper sharpness, perfect without deformations."
)
SUPIR_NEGATIVE = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, "
    "blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
    "deformed, lowres, over-smooth"
)
SEED = 0


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, class_type, **inputs):
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid


def up16(v):
    return (v + 15) // 16 * 16


def lanczos(g, image, w, h):
    return g.add("ImageScale", image=image, upscale_method="lanczos", width=w, height=h, crop="disabled")


def upscale_model(name):
    def build(g, lr, w, h):
        m = g.add("UpscaleModelLoader", model_name=name)
        return g.add("ImageUpscaleWithModel", upscale_model=[m, 0], image=lr)

    return build


nomos2 = upscale_model("4xNomos2_hq_dat2.safetensors")


def seedvr2(unet):
    def build(g, lr, w, h):
        model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
        vae = g.add("VAELoader", vae_name="seedvr2_ema_vae_fp16.safetensors")
        rs = g.add(
            "ImageResizeKJv2",
            image=lr,
            width=w,
            height=h,
            upscale_method="lanczos",
            keep_proportion="stretch",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=8,
            device="cpu",
        )
        pre = g.add("SeedVR2Preprocess", resized_images=[rs, 0])
        tiles = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
        enc = g.add("VAEEncodeTiled", pixels=[pre, 0], vae=[vae, 0], **tiles)
        cond = g.add("SeedVR2Conditioning", model=[model, 0], vae_conditioning=[enc, 0])
        s = g.add(
            "KSampler",
            model=[model, 0],
            positive=[cond, 0],
            negative=[cond, 1],
            latent_image=[enc, 0],
            seed=SEED,
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        )
        dec = g.add("VAEDecodeTiled", samples=[s, 0], vae=[vae, 0], **tiles)
        return g.add(
            "SeedVR2PostProcessing", images=[dec, 0], original_resized_images=[rs, 0], color_correction_method="lab"
        )

    return build


def vosr2(g, lr, w, h):
    m = g.add("VOSR2ModelLoader", model="VOSR2", dtype="default")
    return g.add(
        "VOSR2Upscale",
        model=[m, 0],
        image=lr,
        upscale=4,
        seed=SEED,
        color_alignment="wavelet",
        tile_size=512,
        tile_overlap=64,
        vae_tile_size=0,
        vae_tile_overlap=32,
    )


def iris(g, lr, w, h):
    m = g.add("IrisUpscalerLoader", model_name="iris-3b-upscaler.safetensors", weight_dtype="bf16")
    return g.add("IrisUpscale", upscaler=[m, 0], image=lr, scale=4.0, limit_input_size=False, color_fix=True)


def klein(unet, clip):
    """FLUX.2 klein distilled image edit: 4 steps, cfg 1, the upscaled image as reference latent."""

    def build(g, lr, w, h):
        W, H = up16(w), up16(h)
        model = g.add("UNETLoader", unet_name=unet, weight_dtype="default")
        te = g.add("CLIPLoader", clip_name=clip, type="flux2", device="default")
        vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
        pos = g.add("CLIPTextEncode", clip=[te, 0], text=PROMPT)
        neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
        ref = g.add("VAEEncode", pixels=[lanczos(g, lr, W, H), 0], vae=[vae, 0])
        pos = g.add("ReferenceLatent", conditioning=[pos, 0], latent=[ref, 0])
        neg = g.add("ReferenceLatent", conditioning=[neg, 0], latent=[ref, 0])
        guider = g.add("CFGGuider", model=[model, 0], positive=[pos, 0], negative=[neg, 0], cfg=1.0)
        s = g.add(
            "SamplerCustomAdvanced",
            noise=[g.add("RandomNoise", noise_seed=SEED), 0],
            guider=[guider, 0],
            sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
            sigmas=[g.add("Flux2Scheduler", steps=4, width=W, height=H), 0],
            latent_image=[g.add("EmptyFlux2LatentImage", width=W, height=H, batch_size=1), 0],
        )
        return g.add("VAEDecode", samples=[s, 0], vae=[vae, 0])

    return build


def flux3(g, lr, w, h):
    """FLUX.3 klein image edit: 16 steps, no CFG, alpha 16, prompt upsampler off."""
    W, H = up16(w), up16(h)
    model = g.add("UNETLoader", unet_name="flux3-image-klein-bf16.safetensors", weight_dtype="default")
    model = g.add("ModelSamplingFlux", model=[model, 0], max_shift=2.77259, base_shift=2.77259, width=W, height=H)
    te = g.add("CLIPLoader", clip_name="qwen3vl_4b_bf16.safetensors", type="flux3", device="default")
    vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
    pos = g.add("CLIPTextEncode", clip=[te, 0], text=PROMPT)
    neg = g.add("CLIPTextEncode", clip=[te, 0], text="")
    ref = g.add(
        "ReferenceMultiLatent",
        positive=[pos, 0],
        negative=[neg, 0],
        vae=[vae, 0],
        resolution="keep resolution",
        **{"images.image0": [lanczos(g, lr, W, H), 0]},
    )
    s = g.add(
        "KSampler",
        model=[model, 0],
        positive=[ref, 0],
        negative=[ref, 1],
        latent_image=[g.add("EmptyFlux2LatentImage", width=W, height=H, batch_size=1), 0],
        seed=SEED,
        steps=16,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
    )
    return g.add("VAEDecode", samples=[s, 0], vae=[vae, 0])


def qwen21(g, lr, w, h):
    """Qwen-Image 2.1 image edit: 25 steps, cfg 1, prompt enhancer off, reference kept at full size."""
    W, H = up16(w), up16(h)
    model = g.add("UNETLoader", unet_name="qwen_image_2.1_bf16.safetensors", weight_dtype="default")
    model = g.add("QwenImage21Cache", model=[model, 0], device="auto", dtype="default")
    te = g.add("CLIPLoader", clip_name="qwen3vl_8b_bf16.safetensors", type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
    cond = g.add(
        "TextEncodeQwenImage21",
        clip=[te, 0],
        prompt=PROMPT,
        negative_prompt="",
        resolution=0,
        vae=[vae, 0],
        **{"images.image_1": [lanczos(g, lr, W, H), 0]},
    )
    s = g.add(
        "KSampler",
        model=[model, 0],
        positive=[cond, 0],
        negative=[cond, 1],
        latent_image=[g.add("EmptyLatentImage", width=W, height=H, batch_size=1), 0],
        seed=SEED,
        steps=25,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
    )
    return g.add("VAEDecode", samples=[s, 0], vae=[vae, 0])


def supir(g, lr, w, h):
    """Native SUPIR template: Juggernaut XL v9 Lightning, 10 steps, cfg 1.5, colour transfer back to the input."""
    ckpt = g.add("CheckpointLoaderSimple", ckpt_name="juggernautXL_v9Rdphoto2Lightning.safetensors")
    patch = g.add("ModelPatchLoader", name="SUPIR-v0Q_fp16.safetensors")
    src = lanczos(g, lr, w, h)
    model = g.add(
        "SUPIRApply",
        model=[ckpt, 0],
        model_patch=[patch, 0],
        vae=[ckpt, 2],
        image=[src, 0],
        strength_start=1.0,
        strength_end=0.9,
        restore_cfg=4.0,
        restore_cfg_s_tmin=0.05,
    )
    sdxl = {"width": 1024, "height": 1024, "crop_w": 0, "crop_h": 0, "target_width": 1024, "target_height": 1024}
    pos = g.add("CLIPTextEncodeSDXL", clip=[ckpt, 1], text_g=SUPIR_POSITIVE, text_l=SUPIR_POSITIVE, **sdxl)
    neg = g.add("CLIPTextEncodeSDXL", clip=[ckpt, 1], text_g=SUPIR_NEGATIVE, text_l=SUPIR_NEGATIVE, **sdxl)
    s = g.add(
        "SamplerCustom",
        model=[model, 0],
        add_noise=True,
        noise_seed=SEED,
        cfg=1.5,
        positive=[pos, 0],
        negative=[neg, 0],
        sampler=[g.add("KSamplerSelect", sampler_name="dpmpp_2m_sde"), 0],
        sigmas=[g.add("BasicScheduler", model=[model, 0], scheduler="sgm_uniform", steps=10, denoise=1.0), 0],
        latent_image=[g.add("EmptyLatentImage", width=w, height=h, batch_size=1), 0],
    )
    dec = g.add("VAEDecode", samples=[s, 0], vae=[ckpt, 2])
    return g.add(
        "ColorTransfer",
        image_target=[dec, 0],
        image_ref=[src, 0],
        method="mkl_lab",
        source_stats="per_frame",
        strength=1.0,
    )


def pid(g, lr, w, h):
    """NVIDIA PiD decoding a FLUX.2 latent of the low-res image at 4x, lcm 4 steps, colour bias correction on."""
    W, H = up16(w), up16(h)
    vae = g.add("VAELoader", vae_name="flux2-vae.safetensors")
    latent = g.add("VAEEncode", pixels=[lanczos(g, lr, W // 4, H // 4), 0], vae=[vae, 0])
    model = g.add("UNETLoader", unet_name="pid_flux2_512_to_2048_4step_bf16.safetensors", weight_dtype="default")
    model = g.add("PiDColorBiasCorrection", model=[model, 0], strength=1.0, backbone="flux2")
    te = g.add("CLIPLoader", clip_name="gemma_2_2b_it_elm_fp8_scaled.safetensors", type="pixeldit", device="default")
    text = g.add("CLIPTextEncode", clip=[te, 0], text="")
    pos = g.add("PiDConditioning", positive=[text, 0], latent=[latent, 0], latent_format="flux", degrade_sigma=0.0)
    s = g.add(
        "SamplerCustom",
        model=[model, 0],
        add_noise=True,
        noise_seed=SEED,
        cfg=1.0,
        positive=[pos, 0],
        negative=[g.add("ConditioningZeroOut", conditioning=[text, 0]), 0],
        sampler=[g.add("KSamplerSelect", sampler_name="lcm"), 0],
        sigmas=[g.add("BasicScheduler", model=[model, 0], scheduler="simple", steps=4, denoise=1.0), 0],
        latent_image=[g.add("EmptyChromaRadianceLatentImage", width=W, height=H, batch_size=1), 0],
    )
    return g.add("VAEDecode", samples=[s, 0], vae=[g.add("VAELoader", vae_name="pixel_space"), 0])


METHODS = {
    "nomos2": nomos2,
    "esrgan": upscale_model("RealESRGAN_x4plus.pth"),
    "svr2": seedvr2("seedvr2_7b_int8_convrot.safetensors"),
    "svr2_sharp": seedvr2("seedvr2_7b_sharp_int8_convrot.safetensors"),
    "svr2_3b": seedvr2("seedvr2_3b_fp16.safetensors"),
    "vosr2": vosr2,
    "iris": iris,
    "klein4b": klein("flux-2-klein-4b.safetensors", "qwen_3_4b.safetensors"),
    "klein9b": klein("flux-2-klein-9b-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
    "flux3": flux3,
    "qwen21": qwen21,
    "supir": supir,
    "pid": pid,
}


def build(method, image, w, h, prefix):
    g = Graph()
    lr = [g.add("LoadImage", image=image), 0]
    out = METHODS[method](g, lr, w, h)
    g.add("SaveImage", images=[out, 0], filename_prefix=prefix)
    return g.nodes
