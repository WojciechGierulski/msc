conda create -n msc python=3.10
sudo apt-get install unrar-free
conda activate msc
pip install torch torchvision torchaudio
pip install wandb pytorchvideo matplotlib transformers[torch] tqdm patool
