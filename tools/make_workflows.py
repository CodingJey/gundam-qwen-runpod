#!/usr/bin/env python3
"""Generate the ComfyUI workflow JSON files shipped in workflows/.

Run from the repository root:
    python3 tools/make_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "workflows"

MODEL = "qwen-image-Q4_K_M.gguf"
CLIP = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
VAE = "qwen_image_vae.safetensors"
SPEED_LORA = "Qwen-Image-Lightning-8steps-V1.0.safetensors"
STYLE_LORA = "gundam_qwen_image.safetensors"

PROMPT = (
    "Mobile Suit Gundam mecha, white and blue armor with red and yellow accents, "
    "glowing green camera eyes, V-fin antenna, precise panel lines and decals, "
    "battle damage and weathering, standing in a ruined space colony hangar, "
    "sparks, smoke, volumetric light, cinematic low-angle shot, realistic "
    "sci-fi concept art, incredibly detailed, sharp focus, masterpiece"
)
NEGATIVE = (
    "low quality, blurry, deformed, extra limbs, bad anatomy, watermark, "
    "signature, jpeg artifacts, toy, plastic"
)


class Graph:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.links: list[list[Any]] = []
        self.by_id: dict[int, dict[str, Any]] = {}
        self.next_id = 1
        self.next_link = 1

    def node(
        self,
        class_type: str,
        widgets: list[Any],
        inputs: list[tuple[str, str]] | None = None,
        outputs: list[tuple[str, str]] | None = None,
        *,
        pos: tuple[int, int] = (0, 0),
        size: tuple[int, int] = (320, 120),
        title: str | None = None,
        mode: int = 0,
        note: bool = False,
    ) -> dict[str, Any]:
        node_id = self.next_id
        self.next_id += 1
        node: dict[str, Any] = {
            "id": node_id,
            "type": class_type,
            "pos": [pos[0], pos[1]],
            "size": [size[0], size[1]],
            "flags": {},
            "order": len(self.nodes),
            "mode": mode,
            "inputs": [
                {"name": name, "type": input_type, "link": None}
                for name, input_type in (inputs or [])
            ],
            "outputs": [
                {
                    "name": name,
                    "type": output_type,
                    "slot_index": index,
                    "links": [],
                }
                for index, (name, output_type) in enumerate(outputs or [])
            ],
            "properties": {} if note else {"Node name for S&R": class_type},
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        self.by_id[node_id] = node
        return node

    def connect(
        self,
        source: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str,
    ) -> None:
        source_slot = next(
            index
            for index, output in enumerate(source["outputs"])
            if output["name"] == output_name
        )
        target_slot = next(
            index
            for index, input_slot in enumerate(target["inputs"])
            if input_slot["name"] == input_name
        )
        if target["inputs"][target_slot]["link"] is not None:
            raise ValueError(
                f"input {input_name!r} on node {target['id']} is already connected"
            )
        link_id = self.next_link
        self.next_link += 1
        self.links.append(
            [
                link_id,
                source["id"],
                source_slot,
                target["id"],
                target_slot,
                link_type,
            ]
        )
        target["inputs"][target_slot]["link"] = link_id
        source["outputs"][source_slot]["links"].append(link_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_node_id": self.next_id - 1,
            "last_link_id": self.next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.55, "offset": [180, 280]}},
            "version": 0.4,
        }


def add_model_stack(
    graph: Graph,
    *,
    style_lora: bool,
    speed_lora: bool,
    x: int = 0,
    y: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    unet = graph.node(
        "UnetLoaderGGUF",
        [MODEL],
        outputs=[("MODEL", "MODEL")],
        pos=(x, y),
        size=(340, 110),
    )

    model_source = unet
    if speed_lora:
        speed = graph.node(
            "LoraLoaderModelOnly",
            [SPEED_LORA, 1.0],
            inputs=[("model", "MODEL")],
            outputs=[("MODEL", "MODEL")],
            pos=(x + 380, y),
            size=(360, 140),
            title="Lightning 8-step LoRA",
        )
        graph.connect(model_source, "MODEL", speed, "model", "MODEL")
        model_source = speed

    if style_lora:
        style = graph.node(
            "LoraLoaderModelOnly",
            [STYLE_LORA, 0.85],
            inputs=[("model", "MODEL")],
            outputs=[("MODEL", "MODEL")],
            pos=(x + 380 if not speed_lora else x + 780, y),
            size=(360, 140),
            title="Gundam style LoRA",
        )
        graph.connect(model_source, "MODEL", style, "model", "MODEL")
        model_source = style

    sampling = graph.node(
        "ModelSamplingAuraFlow",
        [3.1],
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        pos=(x + 1080, y),
        size=(300, 110),
    )
    graph.connect(model_source, "MODEL", sampling, "model", "MODEL")

    return unet, sampling, model_source


def add_text_conditioning(
    graph: Graph,
    *,
    x: int,
    y: int,
    prompt: str = PROMPT,
    negative: str = NEGATIVE,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clip = graph.node(
        "CLIPLoaderGGUF",
        [CLIP, "qwen_image"],
        outputs=[("CLIP", "CLIP")],
        pos=(x, y),
        size=(340, 120),
    )
    positive = graph.node(
        "CLIPTextEncode",
        [prompt],
        inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        pos=(x + 380, y),
        size=(430, 230),
        title="Positive prompt",
    )
    negative_node = graph.node(
        "CLIPTextEncode",
        [negative],
        inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        pos=(x + 380, y + 270),
        size=(430, 170),
        title="Negative prompt",
    )
    graph.connect(clip, "CLIP", positive, "clip", "CLIP")
    graph.connect(clip, "CLIP", negative_node, "clip", "CLIP")
    return clip, positive, negative_node


def add_sampler_and_output(
    graph: Graph,
    *,
    model_source: dict[str, Any],
    positive: dict[str, Any],
    negative: dict[str, Any],
    latent: dict[str, Any],
    steps: int,
    cfg: float,
    denoise: float,
    x: int,
    y: int,
    seed: int = 50347169638278,
) -> None:
    ksampler = graph.node(
        "KSampler",
        [seed, "randomize", steps, cfg, "euler", "simple", denoise],
        inputs=[
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ],
        outputs=[("LATENT", "LATENT")],
        pos=(x, y),
        size=(320, 480),
    )
    vae = graph.node(
        "VAELoader",
        [VAE],
        outputs=[("VAE", "VAE")],
        pos=(x, y + 520),
        size=(340, 90),
    )
    decode = graph.node(
        "VAEDecode",
        [],
        inputs=[("samples", "LATENT"), ("vae", "VAE")],
        outputs=[("IMAGE", "IMAGE")],
        pos=(x + 380, y),
        size=(250, 100),
    )
    save = graph.node(
        "SaveImage",
        ["gundam"],
        inputs=[("images", "IMAGE")],
        pos=(x + 380, y + 160),
        size=(600, 600),
    )
    graph.connect(model_source, "MODEL", ksampler, "model", "MODEL")
    graph.connect(positive, "CONDITIONING", ksampler, "positive", "CONDITIONING")
    graph.connect(negative, "CONDITIONING", ksampler, "negative", "CONDITIONING")
    graph.connect(latent, "LATENT", ksampler, "latent_image", "LATENT")
    graph.connect(ksampler, "LATENT", decode, "samples", "LATENT")
    graph.connect(vae, "VAE", decode, "vae", "VAE")
    graph.connect(decode, "IMAGE", save, "images", "IMAGE")


def note(graph: Graph, text: str, *, x: int, y: int, width: int = 720) -> None:
    graph.node(
        "MarkdownNote",
        [text],
        pos=(x, y),
        size=(width, 220),
        title="How to use this workflow",
        note=True,
    )


def text_to_image(*, style_lora: bool) -> dict[str, Any]:
    graph = Graph()
    _, sampling, _ = add_model_stack(graph, style_lora=style_lora, speed_lora=True)
    _, positive, negative = add_text_conditioning(graph, x=0, y=180)
    latent = graph.node(
        "EmptySD3LatentImage",
        [1024, 1024, 1],
        outputs=[("LATENT", "LATENT")],
        pos=(0, 500),
        size=(300, 150),
    )
    add_sampler_and_output(
        graph,
        model_source=sampling,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=8,
        cfg=1.0,
        denoise=1.0,
        x=1080,
        y=180,
    )

    if style_lora:
        text = (
            "## Gundam Qwen-Image (RTX 3090)\n\n"
            "1. The **Lightning 8-step LoRA** is enabled: keep `steps=8`, `cfg=1.0`, "
            "`euler` / `simple` for fast drafts.\n"
            "2. The **Gundam style LoRA** must be named "
            "`gundam_qwen_image.safetensors`. Put its trigger words at the start of "
            "the positive prompt and set strength around `0.8-1.0`.\n"
            "3. For a final image, bypass the Lightning node (`Ctrl+B`), set "
            "`steps=20-30`, `cfg=3.5-4.5`, then queue again.\n"
            "4. `1024x1024` is a safe 3090 size. Qwen's native `1328x1328` is slower; "
            "reduce to `1024x1024` (or lower the GGUF quant) if you hit OOM.\n"
            "5. If you do not have a style LoRA yet, use the `qwen_image_base` workflow."
        )
    else:
        text = (
            "## Qwen-Image base (no style LoRA)\n\n"
            "1. This workflow uses **Qwen-Image Q4_K_M GGUF** + Lightning 8-step LoRA.\n"
            "2. Keep `steps=8`, `cfg=1.0`, `euler` / `simple` for fast drafts; bypass "
            "the Lightning LoRA and use `steps=20-30`, `cfg=3.5-4.5` for a final image.\n"
            "3. Add your Gundam LoRA in the UI: add a `LoraLoaderModelOnly` node between "
            "the model loader and `ModelSamplingAuraFlow`, then select your file in "
            "`models/loras`.\n"
            "4. Start at `1024x1024` on a 3090. Qwen's native `1328x1328` is heavier."
        )
    note(graph, text, x=0, y=-260)
    return graph.to_dict()


def image_to_image(*, style_lora: bool) -> dict[str, Any]:
    graph = Graph()
    _, sampling, _ = add_model_stack(graph, style_lora=style_lora, speed_lora=True)
    _, positive, negative = add_text_conditioning(
        graph,
        x=0,
        y=220,
        prompt=(
            "Refine this Gundam image: keep the same mobile suit design and pose, "
            "add realistic weathering, panel-line detail, hangar lighting, and "
            "cinematic atmosphere, high detail, masterpiece"
        ),
    )
    load_image = graph.node(
        "LoadImage",
        ["gundam_upload_placeholder.png", "image"],
        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
        pos=(0, 760),
        size=(340, 430),
        title="Upload an image to refine",
    )
    scale = graph.node(
        "ImageScale",
        ["lanczos", 1024, 1024, "disabled"],
        inputs=[("image", "IMAGE")],
        outputs=[("IMAGE", "IMAGE")],
        pos=(380, 760),
        size=(300, 160),
        title="Scale to 1024x1024",
    )
    encode = graph.node(
        "VAEEncode",
        [],
        inputs=[("pixels", "IMAGE"), ("vae", "VAE")],
        outputs=[("LATENT", "LATENT")],
        pos=(720, 760),
        size=(260, 100),
    )
    graph.connect(load_image, "IMAGE", scale, "image", "IMAGE")
    graph.connect(scale, "IMAGE", encode, "pixels", "IMAGE")
    # VAELoader is created inside add_sampler_and_output; connect after it exists.

    add_sampler_and_output(
        graph,
        model_source=sampling,
        positive=positive,
        negative=negative,
        latent=encode,
        steps=8,
        cfg=1.0,
        denoise=0.62,
        x=1080,
        y=220,
    )
    vae = next(node for node in graph.nodes if node["type"] == "VAELoader")
    graph.connect(vae, "VAE", encode, "vae", "VAE")

    lora_note = (
        " The **Gundam style LoRA** is enabled and must be named "
        "`gundam_qwen_image.safetensors`."
        if style_lora
        else " No style LoRA is loaded in this version."
    )
    text = (
        "## Qwen-Image img2img (refine an existing image)\n\n"
        "1. Click **choose file to upload** in `LoadImage` and pick a generated image.\n"
        "2. `denoise=0.62` keeps the composition and changes medium amounts. Use "
        "`0.4-0.5` for subtle cleanup, `0.7-0.8` for a stronger restyle.\n"
        "3. The Lightning LoRA is enabled (`steps=8`, `cfg=1.0`). For a final pass, "
        "bypass it and use `steps=20-30`, `cfg=3.5-4.5`.\n"
        "4. The input is scaled to `1024x1024`. Change `ImageScale` if you need another "
        "aspect ratio." + lora_note
    )
    note(graph, text, x=0, y=-260)
    return graph.to_dict()


def main() -> None:
    WORKFLOWS.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gundam_qwen_image.json": text_to_image(style_lora=True),
        "qwen_image_base.json": text_to_image(style_lora=False),
        "gundam_qwen_img2img.json": image_to_image(style_lora=True),
        "qwen_image_img2img_base.json": image_to_image(style_lora=False),
    }
    for filename, data in outputs.items():
        path = WORKFLOWS / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({len(data['nodes'])} nodes, {len(data['links'])} links)")


if __name__ == "__main__":
    main()
