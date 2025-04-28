# GNN-Based Materials Property Prediction

## Setting

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html 
pip install -r requirements.txt
```

Generate dataset
```
python data/make_dataset.py --nn_strategy crystal
```

Train model
```
python .\main.py --config .\configs\test.yaml
```