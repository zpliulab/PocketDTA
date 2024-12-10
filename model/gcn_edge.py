# 基于gnn_node.py设计的有边信息的模型,这里的图卷积聚合是自己写的，基于GCNConv
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, Sequential, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.nn import inits, MessagePassing



class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.aggr = 'maen'
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        
        return edge_weight * x_j


 # “Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification”transformer图卷积


class protein_gnn(nn.Module):
    def __init__(self, act, len_pos_embeding, len_feature_edge, hidden_channels=128, output_channels=128, dropout=0.2):
        super(protein_gnn, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        # self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.act = act
        self.lin_x1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_x2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_pos1 = nn.Linear(len_pos_embeding, hidden_channels)
        self.lin_pos2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_edge1 = nn.Linear(len_feature_edge, hidden_channels)
        self.lin_edge2 = nn.Linear(hidden_channels, hidden_channels)
        self.lins_cat = nn.ModuleList([nn.Linear(hidden_channels*2, hidden_channels)])
        self.lins_cat.append(nn.Linear(hidden_channels, hidden_channels))
        self.final = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, t_pos_embeding, t_feature_edge, t_edge_index):
        x_lin_1 = self.act(self.lin_x1(x))
        x_lin_2 = self.act(self.lin_x2(x))

        feature0 = self.lin_pos1(t_pos_embeding)
        h0 = self.conv0(x_lin_1, t_edge_index, feature0)
        h0 = self.act(self.lin_pos2(h0))
        h0 = self.dropout(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_edge1(t_feature_edge)
        h1 = self.conv1(x_lin_1, t_edge_index, feature1)
        h1 = self.act(self.lin_edge2(h1))
        h1 = self.dropout(h1)
        h1 = self.dropout(h1)

        h = torch.cat((h0, h1),1)
        for lin in self.lins_cat:
            h = self.act(lin(h)) 

        h = h + x_lin_2
        # h = self.lin_resnet(h)
        h = self.final(h)
        return h


class protein_conv(nn.Module):
    def __init__(self, len_pc_feature, len_hmm_feature, len_bb_feature, len_pos_embeding, len_feature_edge, hidden_channels, act, dropout, num_blocks=2):
        super(protein_conv, self).__init__()
        self.lin_pc = nn.Linear(len_pc_feature, hidden_channels)
        self.lin_hmm = nn.Linear(len_hmm_feature, hidden_channels)
        self.lin_bb = nn.Linear(len_bb_feature, hidden_channels)
        # self.len_feature_edge
        self.lin_x = nn.Linear(hidden_channels, hidden_channels)
        self.lin_edge_feature = nn.Linear(len_pos_embeding+len_feature_edge, hidden_channels)
        self.act = act
        self.protein_gnn_blocks = nn.ModuleList([
            protein_gnn(act=act,
                        len_pos_embeding=len_pos_embeding,
                        len_feature_edge=len_feature_edge,
                        hidden_channels = hidden_channels, 
                        output_channels=hidden_channels,
                        dropout=dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # 对GCN的尝试
        # 自定义的transformer图卷积
        def define_layers(hidden_channels):
            gcn_layer_sizes = [hidden_channels,hidden_channels]
            layers = []
            for i in range(len(gcn_layer_sizes) - 1):            
                layers.append((
                    TransformerConv(
                        gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=hidden_channels),
                    'x, edge_index, edge_attr -> x'
                ))
                layers.append(nn.LeakyReLU())
                return layers

        self.gcn_protein0 = Sequential(
            'x, edge_index, edge_attr', define_layers(hidden_channels))
        self.gcn_protein1 = Sequential(
            'x, edge_index, edge_attr', define_layers(hidden_channels))
        self.gcn_protein2 = Sequential(
            'x, edge_index, edge_attr', define_layers(hidden_channels))
        
        # self.gcn_protein0 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        # self.gcn_protein1 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        # self.gcn_protein2 = GCNConv(in_channels = hidden_channels,out_channels = hidden_channels)
        
    def forward(self, data_pro):
        # target_x, target_edge_weights, target_edge_index, target_batch = data_pro.x[:,:-6], data_pro.edge_weights, data_pro.edge_index, data_pro.batch
        t_pc_feature, seq_pc_feature, t_pos_embeding, t_feature_edge, t_edge_index= data_pro.x, data_pro.seq_pc_feature, data_pro.t_pos_embeding, data_pro.t_feature_edge, data_pro.edge_index
        
        # t_edge_index,_ = add_self_loops(t_edge_index, num_nodes=len(data_pro))
        t_batch = data_pro.batch
        t_pc_feature = self.act(self.lin_pc(t_pc_feature.to(torch.float32)))
        pro_encode = self.act(self.lin_x(t_pc_feature))  # 这里按理说应该都是在最初特征上分别做两次线性变换，而不是在一次变换的基础上再做一次变换
        for protein_gnn_block in self.protein_gnn_blocks:
            pro_encode = protein_gnn_block(pro_encode, t_pos_embeding, t_feature_edge, t_edge_index) + pro_encode
        
        pro_encode = torch.cat((t_pc_feature, pro_encode),1)

        # transformergcn尝试
        # edge_feature = self.act(self.lin_edge_feature(torch.cat((t_pos_embeding, t_feature_edge),1)))
        # pro_encode = self.act(self.gcn_protein0(t_pc_feature,t_edge_index, edge_feature)) + t_pc_feature
        # pro_encode = self.act(self.gcn_protein1(pro_encode,t_edge_index, edge_feature)) + pro_encode
        # pro_encode = self.act(self.gcn_protein2(pro_encode,t_edge_index, edge_feature)) + pro_encode
        # pro_encode = torch.cat((t_pc_feature, pro_encode),1)

        return pro_encode


class drug_gnn(nn.Module):
    def __init__(self, hidden_channel, output_channel, len_edge_attr, len_edge_rbf,act,dropout):
        super(drug_gnn, self).__init__()
        self.conv0 = EdgeGraphConv(hidden_channel, hidden_channel)
        self.conv1 = EdgeGraphConv(hidden_channel, hidden_channel)
        # self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.lin_x1 = nn.Linear(hidden_channel, hidden_channel)
        self.lin_x2 = nn.Linear(hidden_channel, hidden_channel)
        self.lin_attr1 = nn.Linear(len_edge_attr, hidden_channel)
        self.lin_attr2 = nn.Linear(len_edge_attr, hidden_channel)
        self.lin_rbf1 = nn.Linear(len_edge_rbf, hidden_channel)
        self.lin_rbf2= nn.Linear(hidden_channel, hidden_channel)
        self.lins_cat = nn.ModuleList([nn.Linear(hidden_channel*2, hidden_channel)])
        self.lins_cat.append(nn.Linear(hidden_channel, hidden_channel))
        self.final = nn.Linear(hidden_channel, output_channel)
        self.act = act
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_rbf):
        x1 = self.act(self.lin_x1(x))
        x2 = self.act(self.lin_x2(x))

        # edge_attr = self.lin_attr1(edge_attr)
        # h0 = self.conv0(x, edge_index, edge_attr)
        # h0 = self.act(self.lin_attr2(h0))
        # h0 = self.dropout(h0)

        edge_rbf = self.lin_rbf1(edge_rbf)
        h1 = self.conv1(x1, edge_index, edge_rbf)
        h1 = self.act(self.lin_rbf2(h1))
        h1 = self.dropout(h1)

        # h = torch.cat((h0, h1),1)
        # for lin in self.lins_cat:
        #     h = self.act(lin(h))

        h = h1 + x2
        # h = self.lin_resnet(h)
        h = self.final(h)
        return h

class drug_conv(nn.Module):
    def __init__(self, hidden_channel, len_feature_v, len_edge_attr, len_edge_rbf, act, dropout, num_blocks=2):
        super(drug_conv,self).__init__()
        self.lin_v1 = nn.Linear(len_feature_v, hidden_channel)
        self.lin_v2= nn.Linear(len_feature_v, hidden_channel)
        self.lin_attr = nn.Linear(len_edge_attr, hidden_channel)
        self.lin_rbf = nn.Linear(len_edge_rbf, hidden_channel)
        self.lin_edge_feature = nn.Linear(len_edge_attr+len_edge_rbf, hidden_channel)
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.drug_gnn_blocks = nn.ModuleList([
            drug_gnn(hidden_channel = hidden_channel,
                    output_channel = hidden_channel, 
                    len_edge_attr = len_edge_attr,
                    len_edge_rbf = len_edge_rbf,
                    act = act,
                    dropout = dropout) for _ in range(num_blocks)
        ])

        # 对GCN的尝试
        # 自定义的transformer图卷积
        def define_layers(hidden_channels):
            gcn_layer_sizes = [hidden_channels,hidden_channels]
            layers = []
            for i in range(len(gcn_layer_sizes) - 1):            
                layers.append((
                    TransformerConv(
                        gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=hidden_channels),
                    'x, edge_index, edge_attr -> x'
                ))
                layers.append(nn.LeakyReLU())
                return layers

        # self.gcn_drug0 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channel))
        # self.gcn_drug1 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channel))
        # self.gcn_drug2 = Sequential(
        #     'x, edge_index, edge_attr', define_layers(hidden_channel))
        self.gcn_drug0 = GCNConv(in_channels = hidden_channel,out_channels = hidden_channel)
        self.gcn_drug1 = GCNConv(in_channels = hidden_channel,out_channels = hidden_channel)
        self.gcn_drug2 = GCNConv(in_channels = hidden_channel,out_channels = hidden_channel)

    def forward(self, drug):
        feature_v, seq_features_v, edge_index, edge_attr, edge_rbf = drug.x, drug.seq_features_v, drug.edge_index, drug.edge_attrs, drug.edge_rbf
        
        d_encode1 = self.act(self.lin_v1(feature_v))
        d_encode = self.act(self.lin_v2(feature_v))
        for drug_gnn_block in self.drug_gnn_blocks:
            d_encode = drug_gnn_block(d_encode, edge_index, edge_attr, edge_rbf) + d_encode

        d_encode = torch.cat((d_encode1, d_encode),1)

        # transformergcn的尝试
        # edge_features = self.act(self.lin_rbf(edge_rbf))
        # feature_v = self.act(self.lin_v1(feature_v))
        # d_encode = self.act(self.gcn_drug0(feature_v, edge_index, edge_features)) + feature_v
        # d_encode = self.act(self.gcn_drug1(d_encode,edge_index, edge_features)) + d_encode
        # d_encode = self.act(self.gcn_drug2(d_encode,edge_index, edge_features)) + d_encode
        # d_encode = torch.cat((feature_v,d_encode),1)

        # feature_v = self.act(self.lin_v1(seq_features_v))
        # d_encode = self.act(self.gcn_drug0(feature_v, edge_index)) + feature_v
        # d_encode = self.act(self.gcn_drug1(d_encode,edge_index)) + d_encode
        # d_encode = self.act(self.gcn_drug2(d_encode,edge_index)) + d_encode
        # d_encode = torch.cat((feature_v, d_encode),1)

        return d_encode


def swish(x):
    return x * torch.sigmoid(x)

# 原本药物部分的特征维度：num_features_mol=78, num_edge_feature_mol=1
# 蛋白质33
# GCN based model
class GCN_Edge(nn.Module):
    def __init__(self, len_drug_feature_v=300, len_drug_edge_attr=2, len_drug_edge_rbf=16,
                 len_pc_feature=1024, len_hmm_feature=30, len_bb_feature=6, len_pos_embeding=16, len_feature_edge=26, num_blocks=3,
                 hidden_channels=256, dropout=0,
                 device='cuda:0',
                 n_output=1):
        super(GCN_Edge, self).__init__()

        self.act = nn.LeakyReLU()
        # 药物部分
        """
        self.n_output = n_output
        self.mol_lin_node = nn.Linear(num_features_mol, 128)
        self.mol_lin_edge = nn.Linear(num_edge_feature_mol, num_edge_feature_mol)
        # self.mol_conv = EdgeGraphConv(128, 128)
        self.mol_conv1 = GCNConv(128, 128)
        self.mol_conv2 = GCNConv(128, 128)
        self.mol_conv3 = GCNConv(128, 128)
        self.mol_fc_g1 = nn.Linear(128*2, 256)
        # self.mol_fc_g2 = Linear(256, 128)
        """
        self.drug = drug_conv(hidden_channel = hidden_channels, 
                                len_feature_v = len_drug_feature_v, 
                                len_edge_attr = len_drug_edge_attr, 
                                len_edge_rbf = len_drug_edge_rbf, 
                                act = self.act, 
                                dropout = dropout, 
                                num_blocks=num_blocks)
        self.lin_drug1 = nn.Linear(hidden_channels*2, hidden_channels*2)
        self.lin_drug2 = nn.Linear(hidden_channels*2, hidden_channels*2)

        # 蛋白质部分
        self.pronet = protein_conv(len_pc_feature=len_pc_feature, 
                                   len_hmm_feature=len_hmm_feature, 
                                   len_bb_feature=len_bb_feature,
                                   len_pos_embeding=len_pos_embeding,
                                   len_feature_edge=len_feature_edge,
                                   hidden_channels=hidden_channels,
                                   num_blocks=num_blocks, 
                                   dropout=dropout, 
                                   act=self.act)
        self.lin_pro1 = nn.Linear(hidden_channels*2, hidden_channels*2)
        self.lin_pro2 = nn.Linear(hidden_channels*2, hidden_channels*2)
        
        # 拼接
        self.fc1 = nn.Linear(hidden_channels*4, hidden_channels*4)
        self.fc2 = nn.Linear(hidden_channels*4, hidden_channels*2)
        self.final = nn.Linear(hidden_channels*2, n_output)
        self.dropout = nn.Dropout(dropout)
        self.layernorm_drug = nn.LayerNorm(normalized_shape=hidden_channels*2, elementwise_affine=True)
        self.layernorm_protein = nn.LayerNorm(normalized_shape=hidden_channels*2, elementwise_affine=True)
        self.layernorm = nn.LayerNorm(normalized_shape=hidden_channels*4, elementwise_affine=True)
        self.norm1d = nn.BatchNorm1d(hidden_channels*4)
        # self.norm1d = nn.BatchNorm1d(hidden_channels*4)

    def forward(self, data_mol, data_pro):
        # drug
        """
        # get graph input
        mol_x, mol_edge_index, mol_edge_attrs, mol_edge_rbf, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_rbf, data_mol.batch
        

        # mol_x = self.mol_lin_node(mol_x)
        # # x1 = self.mol_conv(mol_x, mol_edge_index, mol_edge_weights)
        # x1 = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weights)
        # x1 = self.act(x1) + mol_x

        # x2 = self.mol_conv2(x1, mol_edge_index, mol_edge_weights)
        # x2 = self.act(x2) + x1

        # x3 = self.mol_conv3(x2, mol_edge_index, mol_edge_weights)
        # x3 = self.act(x3) + x2
        
        # # x4 = self.mol_conv(x3, mol_edge_index, mol_edge_weights)
        # # x4 = self.relu(x4) + x3
        
        # # 1,3层残差
        # x1 = gep(x1, mol_batch)  # global pooling
        # x3 = gep(x3, mol_batch)  # global pooling
        # x = torch.cat((x1,x3),1)
        # # flatten
        # x = self.act(self.mol_fc_g1(x))
        # # x = self.dropout(x)
        """
        drug_encode = self.layernorm_drug(self.drug(data_mol))
        drug_encode = gep(drug_encode, data_mol.batch)
        drug_encode = self.act(self.lin_drug1(drug_encode))
        drug_encode = self.act(self.lin_drug2(drug_encode))

        # pro
        protein_encode = self.layernorm_protein(self.pronet(data_pro))
        protein_encode = gep(protein_encode, data_pro.batch)
        protein_encode = self.act(self.lin_pro1(protein_encode))
        protein_encode = self.act(self.lin_pro2(protein_encode))

        # 拼接pro和drug
        xc = self.layernorm(torch.cat((drug_encode, protein_encode),1))
        
        # add some dense layers
        xc = self.act(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.act(self.fc2(xc))
        xc = self.dropout(xc)
        out = self.final(xc)

        return out

