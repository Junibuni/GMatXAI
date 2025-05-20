@echo off

echo [0/5] Creating virtual environment...
python -m venv venv
call .\venv\Scripts\activate

echo [1/5] Installing from requirements.txt...
pip install -r requirements.txt

echo [2/5] Installing PyTorch (%CUDA_VERSION%)...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo [3/5] Installing torch_geometric...
pip install torch_geometric

echo [4/5] Installing PyG dependencies...
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

echo Setup complete with CUDA version: %CUDA_VERSION%
