conda create -n msc python=3.10
sudo apt-get install unrar-free
conda activate msc
pip install torch torchvision torchaudio
pip install wandb pytorchvideo matplotlib transformers[torch] tqdm patool
cd /usr/local/cuda-12.1/lib64
sudo rm -f libcudnn*
cd /usr/local/cuda-12.1/include
sudo rm -f cudnn*
nano /opt/conda/envs/msc/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py
