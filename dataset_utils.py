import torch
import math

from pathlib import Path
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    DBLP,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
    Flickr,
)
        
def DataLoader(name):
    name = name.lower()
    normalized_name = name.replace('-', '_')
    data_root = Path(__file__).resolve().parent / 'data'

    if name in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(data_root, name)
        dataset = Planetoid(path, name, transform=None)
    elif name in ['cs', 'physics']:
        path = osp.join(data_root, name)
        dataset = Coauthor(path, name, transform=None)
    elif name in ['computers', 'photo']:
        path = osp.join(data_root, name)
        dataset = Amazon(path, name, transform=None)
    elif name in ['chameleon', 'crocodile', 'squirrel']:
         path = osp.join(data_root, name)
         dataset = WikipediaNetwork(path, name, transform=None)
    elif name in ['cornell', 'texas', 'wisconsin']:
        path = osp.join(data_root, name)
        dataset = WebKB(path, name)
    elif normalized_name in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
        path = osp.join(data_root, normalized_name)
        dataset = HeterophilousGraphDataset(path, normalized_name, transform=None)
    elif name in ['dblp']:
        path = osp.join(data_root, name)
        dataset = DBLP(path, name)
    elif name == 'actor':  # Custom Actor dataset
        path = osp.join(data_root, name)
        dataset = Actor(path)
    elif name == 'flickr': # <--- 第2处修改：加上 Flickr 的加载逻辑
        path = osp.join(data_root, name)
        dataset = Flickr(path, transform=None)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')


    return dataset
