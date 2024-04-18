from flask import Flask, template_rendered, redirect, send_file, render_template_string
import base64
from modal import Image, Stub
app = Flask(__name__)
image = Image.debian_slim().pip_install("gradio")
stub = Stub()


import io
from pathlib import Path
from modal import (
    Image,
    Mount,
    Stub,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)
sdxl_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.26.3",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
        "Flask",
    )
)

# stub = Stub("stable-diffusion-xl")

with sdxl_image.imports():
    from flask import Flask, template_rendered, redirect
    import torch
    from diffusers import DiffusionPipeline
    from fastapi import Response
    from huggingface_hub import snapshot_download

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=2, image=sdxl_image)
class Model:
    @build()
    def build(self):
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        ]
        snapshot_download(
            "RunDiffusion/Juggernaut-XL-v9", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    # @method()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v9", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            width=1024,
            height=1024,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt, n_steps=35, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()

# @stub.cls(gpu=gpu.T4(), container_idle_timeout=2, image=sdxl_image)

def home_handle(mdl):
    for x in range (5):
        image_bytes = mdl.inference.remote(" cinematic portrait of a wood sculpture of a cat")
    if image_bytes:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        # Embed base64-encoded string into HTML image tag
        html_content = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Image">'
        return render_template_string(html_content)
        # return send_file(image_bytes, mimetype='image/jpeg')
        # return f"{image_bytes}"
    else:
        return "Not yo gango"    

def load_mdl():
    model_loaded = Model().enter()
    # model_loaded = Model().enter().remote()
    if model_loaded:
        return "loaded"
    else:
        return f"couldnt load{model_loaded}"

@stub.function(image=image)
@app.route('/api/generate')
def home():
    my_modal = Model()
    # my_modal.build()
    return home_handle(my_modal)

@stub.function(image=image)
@app.route('/api/loadm')
def loadm():
    return load_mdl()

@stub.local_entrypoint()
def main():
    app.run(host="0.0.0.0", port="8080", threaded=True)
