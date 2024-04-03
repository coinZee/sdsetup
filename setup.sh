#!/bin/bash

cd /root/

apt install wget curl -y

directory="/root/stable-diffusion-webui-forge/"

if [ -d "$directory" ]; then
    echo "exist doing it"
    python /root/stable-diffusion-webui-forge/launch.py --enable-insecure-extension-access --share --xformers
else
    echo "Directory does not exist."
    git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git

    cd stable-diffusion-webui-forge/
    cd extensions/
    current_dir=$(pwd)
    echo "Current directory: $current_dir"
    echo "Downloading Extensions"
    git clone https://github.com/BlafKing/sd-civitai-browser-plus.git && \
    git clone https://github.com/Bing-su/adetailer.git

    cd /root/stable-diffusion-webui-forge/models/Stable-diffusion/
    current_dir=$(pwd)
    echo "Current directory: $current_dir"
    wget "https://civitai.com/api/download/models/357609" --content-disposition
    wget "https://civitai.com/api/download/models/348913" --content-disposition
    wget "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors" --content-disposition

    cd /root/stable-diffusion-webui-forge/
    pip3 install -U xformers==0.0.16 opencv-python-headless
    python launch.py --enable-insecure-extension-access --share --xformers 

fi

