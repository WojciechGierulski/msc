conda create -n msc python=3.10
sudo apt-get install unrar-free
conda activate msc

# linux
pip install torch torchvision torchaudio
# windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install "numpy<2"
pip install wandb matplotlib transformers[torch] tqdm patool
pip install git+https://github.com/facebookresearch/pytorchvideo

# optional for old environment
# cd /usr/local/cuda-12.1/lib64
# sudo rm -f libcudnn*
# cd /usr/local/cuda-12.1/include
# sudo rm -f cudnn*