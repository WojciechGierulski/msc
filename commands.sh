conda create -n msc python=3.10
sudo apt-get install unrar-free
conda activate msc
# linux
# pip install torch torchvision torchaudio
# windows
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2"
pip install wandb pytorchvideo matplotlib transformers[torch] tqdm patool
