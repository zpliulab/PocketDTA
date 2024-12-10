import numpy as np
import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import pandas as pd
from protein2graph import Pdb2graph
from ligand2graph import ligand2graph
from ligand_pretrain_models.ligand_pretrain_gnns import GNN_graphpred
# 导入transprot的内容
from transformers import BertModel, BertTokenizer, AlbertModel, AlbertTokenizer, XLNetTokenizer, XLNetModel, T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModel, ElectraTokenizer, ElectraForMaskedLM, ElectraModel
from tqdm import tqdm

def data2graph(relative_path, dataset, data_split, device):
    print("start {} protein and drug embedding".format(data_split))

    model_name = "Rostlab/prot_t5_xl_bfd"
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model_protein = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
        model_protein = AlbertModel.from_pretrained(model_name)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
        model_protein = BertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
        model_protein = XLNetModel.from_pretrained(model_name)
    elif "electra" in model_name:
        tokenizer = ElectraTokenizer(model_name + "/vocab.txt", do_lower_case=False )
        # model_protein = ElectraForMaskedLM.from_pretrained(model_name)
        model_protein = ElectraModel.from_pretrained(model_name)
    else:
        print("Unkown model name")
        assert("wrong name of protein pre-trained model")
    model_protein = model_protein.to(device)
    model_protein = model_protein.eval()

    # 配体预训练编码模型导入
    model_ligand = GNN_graphpred(5, 300, 1, JK = 'last', drop_ratio = 0.5, graph_pooling = 'mean', gnn_type = 'gin')
    model_ligand.from_pretrained('ligand_pretrain_models/supervised_contextpred.pth')
    model_ligand.eval()

    processed_data_file = os.path.join(relative_path, 'processed', dataset + '_' + data_split +'.pt')

    if not os.path.isfile(processed_data_file):
        file_csv = pd.read_csv(os.path.join(relative_path, dataset, data_split + '.csv'))
        
        # 蛋白质处理
        pdb2graph = Pdb2graph(max_num_neighbors=32, tokenizer = tokenizer, large_model = model_protein, device=device, model_name=model_name)  # 由于药物的键就是构建图结构的输入，但是蛋白质是每个蛋白质的每个pocket作为图结构的输入，所以键应该完全独一
        target_graph = {}
        for i, target_path in tqdm(enumerate(file_csv.loc[:,'pdb_3d_path']), desc="Processing pdb paths"):
            if file_csv.loc[i,'uniprot_id'] not in target_graph.keys():  # 每个蛋白质仅一个对应的口袋，计算量不大
                g = pdb2graph._2graph3d(target_path,file_csv.loc[i,'uniprot_id'], cmap_thresh=8)
                target_graph[file_csv.loc[i,'uniprot_id']] = g
        file_prot_keys = list(target_graph.keys())
        del pdb2graph  # 释放实例，占用内存过多
        
        # 药物处理
        smile2graph = ligand2graph(model_ligand)
        smile_graph = {}
        unique_drug_temp = set()
        for i in tqdm(range(len(file_csv)), desc="Processing drug paths"):
            if file_csv.loc[i,'drug_id'] not in unique_drug_temp:
                g = smile2graph._2graph3d2(file_csv.loc[i,'drug_3d_path'])
                smile_graph[file_csv.loc[i,'drug_id']] = g
                unique_drug_temp.add(file_csv.loc[i,'drug_id'])

        if len(smile_graph) == 0 or len(target_graph) == 0:
            raise Exception('no protein or drug, run the script for datasets preparation.')

        file_drugs, file_prot_keys, file_Y = np.asarray(list(file_csv['drug_id'])), np.asarray(file_csv['uniprot_id']), np.asarray(list(file_csv['affinity']))
        file_dataset = DTADataset(root=relative_path, dataset=dataset, data_split=data_split, xd=file_drugs, target_key=file_prot_keys,
                                y=file_Y, smile_graph=smile_graph, target_graph=target_graph)
    else:
        file_dataset = DTADataset(root=relative_path, dataset=dataset, data_split=data_split)

    return file_dataset


# 标准化为图的数据形式
class DTADataset(InMemoryDataset):
    def __init__(self, root='unseen_data', dataset='kdbnet_kiba',
                 data_split='test',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.data_split = data_split

        if os.path.isfile(self.processed_paths[0]):  
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data_mol, self.data_pro = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, target_key, y, smile_graph, target_graph)
            self.data_mol, self.data_pro = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_' + self.data_split+ '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # 药物部分的处理
            drug_size, seq_features_v, feature_v, edge_index, edge_attr, edge_rbf = smile_graph[smiles]
            GCNData_mol = DATA.Data(x=torch.Tensor(feature_v),
                                    seq_features_v=torch.Tensor(seq_features_v),
                                    edge_index=torch.LongTensor(edge_index),
                                    edge_attrs=torch.Tensor(edge_attr),
                                    edge_rbf=torch.Tensor(edge_rbf),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([drug_size]))

            
            # 蛋白质部分的处理
            target_size, seq_pc_feature, t_pc_feature, t_pos_embeding, t_feature_edge, t_edge_index = target_graph[tar_key]
            pronet_input = DATA.Data(x=torch.Tensor(t_pc_feature),
                                    seq_pc_feature=torch.Tensor(seq_pc_feature),
                                    t_pos_embeding=torch.Tensor(t_pos_embeding),
                                    t_feature_edge=torch.Tensor(t_feature_edge),
                                    edge_index=torch.LongTensor(t_edge_index),
                                    y=torch.FloatTensor([labels]))
            pronet_input.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(pronet_input)

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((self.data_mol, self.data_pro), self.processed_paths[0])

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    TRAIN_BATCH_SIZE = batch_size
    loss_sum = 0
    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        loss_sum += loss.item()
    return loss_sum/batch_idx


# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


#prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

