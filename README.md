# GNN-Based Materials Property Prediction

## Setting

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install -r requirements.txt
```

Generate dataset
```
python data/make_dataset.py --nn_strategy voronoi
```