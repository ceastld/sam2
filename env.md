#Env

```bash
conda create -n sam2 python==3.10 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate sam2

pip install -e . -v
pip install opencv-python
pip install decord

ml CUDA/12.1

```


