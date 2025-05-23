# GNN-Based Materials Property Prediction

## Setting
(windows)
```
setup.bat
```

or

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip install -r requirements.txt
```

Generate dataset
```
python data/make_dataset.py --nn_strategy crystal
```
--> deprecated (no need for args)

```
python data/make_dataset.py
```
or download from [link](https://figshare.com/projects/Bulk_and_shear_datasets/165430) (train, val, test)

Train model
```
python .\main.py --config .\configs\test.yaml
```