#!/bin/bash

cd /root/

apt install wget curl -y

git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git

cd stable-diffusion-webui-forge/
cd extensions/
current_dir=$(pwd)
echo "Current directory: $current_dir"
echo "Downloading Extensions"
git clone https://github.com/BlafKing/sd-civitai-browser-plus.git && \
git clone https://github.com/Bing-su/adetailer.git

cd /root/stable-diffusion-webui-forge/models
/Stable-diffusion/
current_dir=$(pwd)
echo "Current directory: $current_dir"
wget "https://civitai.com/api/download/models/357609" --content-disposition
wget "https://civitai.com/api/download/models/348913" --content-disposition
wget "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors" --content-disposition

cd /root/stable-diffusion-webui-forge/
pip3 install -U xformers==0.0.16 opencv-python-headless

python launch.py --enable-nsecure-extension-access --share --xformers 


