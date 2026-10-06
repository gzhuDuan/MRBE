# MRBE-GNN

## Environment

- Python: 3.11.5
- PyTorch: 2.3.1
- PyTorch Geometric: 2.7.0

## Run

```bash
cd MRBE_GNN
python task_curvature_train.py --dataset cora --device cpu --console-log
```

Datasets are loaded from `data/`. Runtime logs are written to `runs/`.
