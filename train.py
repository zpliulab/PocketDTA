import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import pandas as pd
from model.gcn_edge import GCN_Edge
from utils import *
from metrics import *
import time


datasets = ['kdbnet_kiba','kdbnet_davis','pdbbind_refinedset_2019_split']
dataset = datasets[2]
root_path = "unseen_data"
cuda_name = ['cuda:0', 'cuda:1']
print('cuda_name:', cuda_name)


TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

LR = 0.0005
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'model'
results_dir = 'result'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name[0] if USE_CUDA else 'cpu')
model = GCN_Edge(device=device)


model.to(device)
# model.load_state_dict(torch.load('model/model_GCN_Edge_kdbnet_kiba.model'))
model_st = model.__class__.__name__
print("{} Loaded".format(model_st))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
start_time = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))
dataset_start_time = time.time()
print()
test_dataset = data2graph(root_path, dataset, 'test', 'cpu')  # whole_protein_test
train_dataset = data2graph(root_path, dataset, 'train', 'cpu')  # whole_protein_train
valid_dataset = data2graph(root_path, dataset, 'val', 'cpu')  # whole_protein_val

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                            collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False,
                                            collate_fn=collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                            collate_fn=collate)
dataset_end_time = time.time()
# print("preprocess END time: ", time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time())))
print("preprocess dataset speed :", time.strftime("%H-%M-%S", time.gmtime(dataset_end_time-dataset_start_time)))

print()
best_mse = 1000
best_pcc = 0
best_test_mse = 1000
best_epoch = -1
model_file_name = 'model/model_' + model_st + '_' + dataset  + '_' + start_time +'.model'

train_start_time = time.time()
print()
for epoch in range(NUM_EPOCHS):
    loss_epoch = train(model, device, train_loader, optimizer, epoch + 1, TRAIN_BATCH_SIZE)

    # 每一代验证集效果
    print('predicting for valid data')
    val_labels, val_predicts = predicting(model, device, valid_loader)
    val = get_mse(val_labels, val_predicts)
    val_pcc = get_pearson(val_labels, val_predicts)
    val_ci = get_ci(val_labels, val_predicts)
    with open("log/" + str(NUM_EPOCHS) + '_' + dataset + '_val_' + start_time + '.csv','a') as df:
        df.write(','.join(map(str,[epoch,val,val_pcc,val_ci])))
        df.write('\r\n')

    # 每一代测试集效果，看看是否过拟合
    test_labels, test_predicts = predicting(model, device, test_loader)
    test = get_mse(test_labels, test_predicts)
    test_pcc = get_pearson(test_labels, test_predicts)
    test_ci = get_ci(test_labels, test_predicts)
    with open("log/" + str(NUM_EPOCHS) + '_' + dataset + '_test_' + start_time + '.csv','a') as df:
        df.write(','.join(map(str,[epoch,test,test_pcc,test_ci])))
        df.write('\r\n')

    if val <  best_mse:
        best_mse = val
        best_test_mse = test
        best_test_pcc = test_pcc
        best_test_ci = test_ci
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_file_name)
        print('mse improved at epoch ', best_epoch, '; best_val_mse', best_mse, model_st, dataset)
        print('best test datasets results:', best_test_mse, best_test_pcc, best_test_ci)
    else:
        print('No improvement since epoch ', best_epoch, '; best_val_mse', best_mse, model_st, dataset)
        print('best test datasets results:', best_test_mse, best_test_pcc, best_test_ci)

print("preprocess dataset speed :", time.strftime("%H-%M-%S", time.gmtime(dataset_end_time-dataset_start_time)))
train_end_time = time.time()
print("training speed :", time.strftime("%H-%M-%S", time.gmtime(train_end_time-train_start_time)))
print()
        
