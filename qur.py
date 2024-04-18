import io
import time
import threading
from pathlib import Path

# import torch
# from diffusers import DiffusionPipeline
# from fastapi import Response

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
        "torch"
    )
)

stub = Stub("stable-diffusion-xl")


class Model:
    def __init__(self):
        self.base = None
       
        self.refiner = None

    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

    def _inference(self, prompt):
        negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=24,
            denoising_end=0.8,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=24,
            denoising_start=0.8,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt):
        return self._inference(prompt).getvalue()

    @web_endpoint()
    def web_inference(self, prompt):
        return Response(
            content=self._inference(prompt).getvalue(),
            media_type="image/jpeg",
        )

class QueueItem:
    def __init__(self, prompt, id):
        self.prompt = prompt
        self.id = id
        self.image_bytes = None

class Queue:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.model = Model()

    def add_item(self, prompt, id):
        with self.lock:
            self.queue.append(QueueItem(prompt, id))

    def process_queue(self):
        while True:
            with self.lock:
                if not self.queue:
                    return
                item = self.queue.pop(0)

            image_bytes = self.model.inference.remote(item.prompt)
            item.image_bytes = image_bytes
            print(f"Generated image for prompt {item.id} with ID {item.id}")

    def get_image_bytes(self, id):
        with self.lock:
            item = next((x for x in self.queue if x.id == id), None)
            if item is None:
                item = next((x for x in self.queue if x.image_bytes is not None), None)
                if item is None:
                    return None
            return item.image_bytes

@stub.local_entrypoint()
def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
    queue = Queue()
    threading.Thread(target=queue.process_queue, daemon=True).start()

    while True:
        new_prompt = input("Enter a prompt (or type 'q' to quit): ")
        if new_prompt == "q":
            break

        id = int(time.time())
        queue.add_item(new_prompt, id)
        print(f"Added prompt {id} to queue")

        image_bytes = queue.get_image_bytes(id)
        if image_bytes is not None:
            dir = Path("/tmp/stable-diffusion-xl")
            if not dir.exists():
                dir.mkdir(exist_ok=True, parents=True)

            output_path = dir / f"{id}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)