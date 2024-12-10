import pandas as pd
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from torch_cluster import radius_graph
import torch
import torch.nn.functional as F
from transformers import pipeline
import urllib.request
from io import StringIO
from sklearn.neighbors import NearestNeighbors

# nomarliseq_resnamese
def dic_normaliseq_resnamese(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
        'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R','Unknown':'X'}

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']


res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normaliseq_resnamese(res_weight_table)
res_pka_table = dic_normaliseq_resnamese(res_pka_table)
res_pkb_table = dic_normaliseq_resnamese(res_pkb_table)
res_pkx_table = dic_normaliseq_resnamese(res_pkx_table)
res_pl_table = dic_normaliseq_resnamese(res_pl_table)
res_hydrophobic_ph2_table = dic_normaliseq_resnamese(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normaliseq_resnamese(res_hydrophobic_ph7_table)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = 'X'
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# 得到onehot类型的编码，以上部分都是onehot编码的部分
def seq_pc_n_acid_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


# 以下函数都是本论文的编码方式

# 位置编码
def pos_emb(edge_index, num_pos_emb=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_pos_emb)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

# 归一化，并解决可能的nan数据问题
def normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

# 方位矩阵的建立
def quaternions(R):
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rseq_resnamesseq_resnames = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rseq_resnamesseq_resnames,
        - Rxx + Ryy - Rseq_resnamesseq_resnames,
        - Rxx - Ryy + Rseq_resnamesseq_resnames
    ], -1)))
    _R = lambda i,j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyseq_resnames = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyseq_resnames, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

# 建立局部坐标系（边特征）
def local_frame(X, edge_index, eps=1e-6):
    dX = X[1:] - X[:-1]
    U = normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = normalize(torch.cross(u_2, u_1), dim=-1)
    # n_1 = normalize(torch.cross(u_1, u_0), dim=-1)

    o_1 = normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)

    # dX = X[edge_index[0]] - X[edge_index[1]]
    dX = X[edge_index[1]] - X[edge_index[0]]
    dX = normalize(dX, dim=-1)
    # dU = torch.bmm(O[edge_index[1]], dX.unsqueeze(2)).squeeze(2)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])
    Q = quaternions(R)
    O_features = torch.cat((dU,Q), dim=-1)

    return O_features

# rbf函数
def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# 输入为整个蛋白质时，需要用到，用于用pdbid生成蛋白质结构
def get_pdb_structure(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    with urllib.request.urlopen(url) as response:
        pdb_data = response.read().decode('utf-8')
    pdb_io = StringIO(pdb_data)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_io)
    
    return structure


# 仅使用knn确定边
def k_nearest_neighbor(pos, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(pos)
    distances, indices = nn.kneighbors(pos)
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first one since it will be the point itself
            edge_index.append([neighbor,i])
    for i in range(len(edge_index)):
        if [edge_index[i][1], edge_index[i][0]] not in edge_index:
            edge_index.append([edge_index[i][1], edge_index[i][0]])

    edge_index.sort(key=lambda x: (x[1], x[0]))
    edge_index = torch.tensor(np.array(edge_index).T)
            
    return edge_index


# 融合k近邻结果与阈值编码结果
def k_radius_graph(pos, edge_index2, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(pos)
    distances, indices = nn.kneighbors(pos)
    edge_index1 = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first one since it will be the point itself
            edge_index1.append([neighbor,i])
    for i in range(len(edge_index1)):
        if [edge_index1[i][1], edge_index1[i][0]] not in edge_index1:
            edge_index1.append([edge_index1[i][1], edge_index1[i][0]])

    edge_index2 = edge_index2.t().tolist()
    for i in range(len(edge_index1)):
        if edge_index1[i] not in edge_index2:
            edge_index2.append(edge_index1[i])

    edge_index2.sort(key=lambda x: (x[1], x[0]))
    edge_index2 = torch.tensor(np.array(edge_index2).T)
    return edge_index2


# 蛋白质pdb转为图表示
class Pdb2graph():

    def __init__(self, max_num_neighbors, tokenizer, large_model, device, 
                use_large_model='False', model_name = "Rostlab/prot_bert_bfd"):
        self.pdb_structure = None
        self.max_num_neighbors = max_num_neighbors
        self.tokenizer = tokenizer
        self.large_model = large_model
        self.device = device
        self.use_large_model = use_large_model
        self.model_name = model_name

    
    # 蛋白质的完备性表示，仅编码部分，无神经网络聚合部分，包含了边的信息
    def _2graph3d(self, pdbpath, uniprot_id, cmap_thresh=8.0):
        protein = 'pocket'  # 'entire'
        if protein == 'entire':
            # 处理整个蛋白质作为输入的情况，这是做整个蛋白质与蛋白质口袋对比实验室用到的，所有蛋白质数据都是在线获得，没有下载
            pdb_id = uniprot_id  # 输入感兴趣的PDB编号
            pdb_structure = get_pdb_structure(pdb_id)
        else:
            # 蛋白质口袋输入情况
            parser = PDBParser()
            pdb_structure = parser.get_structure(pdbpath, pdbpath)

        # 得到所有Cα的坐标
        pos = []
        for atom in pdb_structure.get_atoms():
            if atom.name == 'CA':
                pos.append(atom.get_coord())
        pos = torch.tensor(pos)

        # 获得蛋白质口袋序列
        residues = [r for r in pdb_structure.get_residues()]
        seq_resnames = [name.resname for name in residues]
        seq = []
        # 得到残差符号表示
        for res in seq_resnames:
            if res in pro_res_table:
                seq.append(res)
            else:
                seq.append('X')

        # 物理化学方法的编码
        seq_pc_feature = seq_pc_n_acid_feature(seq)
        
        # 是否使用大模型，其实也就包含了进化信息
        seq = ' '.join(seq)
        if 'prot_bert_bfd' in self.model_name:
            seq = self.tokenizer(seq, return_tensors='pt')
            seq['attention_mask'] = seq['attention_mask'].to(self.device)
            seq['input_ids'] = seq['input_ids'].to(self.device)
            seq['token_type_ids'] = seq['token_type_ids'].to(self.device)
            pc_feature = self.large_model(**seq).last_hidden_state.squeeze(0)[1:-1,:].detach()
        else:
            fe = pipeline('feature-extraction', model=self.large_model, tokenizer=self.tokenizer, device=self.device)
            embedding = fe(seq)
            embedding = np.array(embedding)
            embedding = embedding.reshape(embedding.shape[1],embedding.shape[2])
            seq_len = len(seq.replace(" ", ""))
            if "t5" in self.model_name:
                start_Idx = 0
                end_Idx = seq_len
                pc_feature = embedding[start_Idx:end_Idx]
            elif "albert" in self.model_name:
                start_Idx = 1
                end_Idx = seq_len+1
                pc_feature = embedding[start_Idx:end_Idx]
            elif "bert" in self.model_name:
                start_Idx = 1
                end_Idx = seq_len+1
                pc_feature = embedding[start_Idx:end_Idx]
            elif "electra" in self.model_name:
                start_Idx = 1
                end_Idx = seq_len+1
                pc_feature = embedding[start_Idx:end_Idx]
            elif "xlnet" in self.model_name:
                padded_seq_len = len(embedding)
                start_Idx = padded_seq_len-seq_len-2
                end_Idx = padded_seq_len-2
                seq_emd = embedding[start_Idx:end_Idx]
                pc_feature.append(seq_emd)
            else:
                assert("none large model name!")
        
        edge_method = 'threshold'  # knn
        if edge_method == 'threshold':
            # 阈值的方法得到边
            edge_index = radius_graph(pos, r=cmap_thresh, max_num_neighbors=self.max_num_neighbors)
        elif edge_method == 'knn':
            # 只使用knn得到边
            edge_index = k_nearest_neighbor((pos, 5))
        else:
            # 融合k近邻结果与阈值得到边
            edge_index2 = radius_graph(pos, r=cmap_thresh, max_num_neighbors=self.max_num_neighbors)
            edge_index = k_radius_graph(pos, edge_index2, 3)
        
        # 空间位置编码
        pos_embeding = pos_emb(edge_index, 16)

        # 结构编码
        E_vectors = pos[edge_index[0]] - pos[edge_index[1]]
        edge_rbf = rbf(E_vectors.norm(dim=-1), D_count=16)
        O_feature = local_frame(pos, edge_index)
        feature_edge = torch.cat([edge_rbf, O_feature, E_vectors], dim=-1)

        # 这里seq_pc_feature特征只在使用物理化学特征代替预训练模型时用到了
        return len(seq_resnames), seq_pc_feature, pc_feature, pos_embeding, feature_edge, edge_index