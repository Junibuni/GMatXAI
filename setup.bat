@echo off

echo Creating virtual environment...
python -m venv venv
call .\venv\Scripts\activate

pip install --upgrade pip wheel setuptools

echo Installing PyTorch (%CUDA_VERSION%)...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo Installing torch_geometric...
pip install torch_geometric

echo Installing PyG dependencies...
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html

echo Installing from requirements.txt...
pip install -r requirements.txt

echo Setup complete with CUDA version: %CUDA_VERSION%
