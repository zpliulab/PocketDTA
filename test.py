import torch
from torch_geometric.data import DataLoader
import pandas as pd
from model.gcn_edge import GCN_Edge
from utils import *
from metrics import *


datasets = ['kdbnet_kiba','kdbnet_davis','pdbbind_refinedset_2019_split']
dataset = datasets[1]
model_path = 'model/model_GCN_Edge_kdbnet_davis_2024-12-05 20-51-37.model'

root_path = "unseen_data"
cuda_name = ['cuda:0', 'cuda:1']
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name[0] if USE_CUDA else 'cpu')
test_dataset = data2graph(root_path, dataset,'test', device)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False,
                                            collate_fn=collate)
best_model = GCN_Edge(device=device).to(device)
best_model.load_state_dict(torch.load(model_path))

test,test_pcc,test_ci = 0,0,0
G, P = predicting(best_model, device, test_loader)
test += get_mse(G, P)
test_pcc += get_pearson(G,P)
test_ci += get_ci(G,P)
print('mse:{}; pcc:{}; ci:{};'.format(test,test_pcc,test_ci))