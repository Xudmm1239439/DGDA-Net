import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
import numpy as np
from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Parameter
from itertools import permutations
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    return node_features

class HypergraphConv(MessagePassing):
    """Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0,  **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.edgefc = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)


    def forward(self, x, hyperedge_index,
                hyperedge_weight = None,
                hyperedge_attr = None, EW_weight = None, dia_len = None):
        """
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)      #[num_nodes]
        D = 1.0 / D                                                         # all 0.5 if hyperedge_weight is None
        D[D == float("inf")] = 0
        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        B = 1.0 / B
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,#hyperedge_attr[hyperedge_index[1]],  
                             size=(num_nodes, num_edges))                   #num_edges,1,100
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_nodes, num_edges))                   #num_nodes,1,100
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        
        return torch.nn.LeakyReLU()(out)  #


    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j      
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
        
class highConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(highConv, self).__init__(aggr='add')  # "Add" aggregation.
        #self.lin = torch.nn.Linear(in_channels, out_channels)
        self.gate = torch.nn.Linear(2*in_channels, 1)
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # x_j e.g.[135090, 512]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #h2 = x_i - x_j
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))#e.g.[135090, 1]

        return norm.view(-1, 1) * (x_j) *alpha_g

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

class PathNN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(PathNN, self).__init__(aggr=aggr)  # Use the specified aggregation method
        self.gate = torch.nn.Linear(2 * in_channels, 1)  # Gate to compute attention weights between nodes
        self.lin = torch.nn.Linear(in_channels, out_channels)  # Linear transformation for node features

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Propagate messages through the graph.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # x_i and x_j are the node features for the source and target nodes, respectively
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Compute the difference and the concatenated features to form the path
        path_features = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(path_features))  # Compute attention or gating for path features

        # The attention weights (alpha_g) control the information flow based on the path
        return norm.view(-1, 1) * x_j * alpha_g  # Weighted message passing based on path features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels], aggregation result
        return self.lin(aggr_out)  # Apply the linear transformation to the aggregated messages


class MaskedNLLLoss(nn.Module):
    
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss
    
class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full', modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(HyperGCN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.act_fn = nn.ReLU()
        self.nhidden = nhidden
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(5000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), highConv(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)

    def forward(self, a, v, l, dia_len, qmask):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)                                     

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i+3):
                if _ < 3:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            
            Gnodes=[]
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*3

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features

class HyperPathNN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full', modals=['a','v','l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(HyperPathNN, self).__init__()
        self.return_feature = return_feature  #True
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.act_fn = nn.ReLU()
        self.nhidden = nhidden
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        #------------------------------------    
        self.fc1 = nn.Linear(n_dim, nhidden)      
        #self.fc2 = nn.Linear(n_dim, nhidden)     
        self.num_L =  num_L
        self.num_K =  num_K
        for ll in range(num_L):
            setattr(self,'hyperconv%d' %(ll+1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(5000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        #nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self,'conv%d' %(kk+1), PathNN(nhidden, nhidden))
        #self.conv = highConv(nhidden, nhidden)

    def forward(self, a, v, l, dia_len, qmask):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(features)  
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item()+1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1*hyperedge_type1 + self.hyperedge_attr2*(1-hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self,'hyperconv%d' %(ll+1))(out, hyperedge_index, weight, edge_attr, EW_weight, dia_len)             
        if self.use_residue:
            out1 = torch.cat([features, out], dim=-1)                                   
        #out1 = self.reverse_features(dia_len, out1)                                     

        #---------------------------------------
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self,'conv%d' %(kk+1))(gnn_out,gnn_edge_index)

        out2 = torch.cat([out,gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)
        out1 = self.reverse_features(dia_len, out2)
        #---------------------------------------
        return out1
        


    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 =[]
        index2 =[]
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i+3):
                if _ < 3:
                    index2 = index2 + [edge_count]*i
                else:
                    index2 = index2 + [edge_count]*3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            
            Gnodes=[]
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_,2))
                tmp = tmp + perm
            batch = batch + [batch_count]*i*3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1]*i + [0]*3

            node_count = node_count + i*num_modality

        index1 = torch.LongTensor(index1).view(1,-1)
        index2 = torch.LongTensor(index2).view(1,-1)
        hyperedge_index = torch.cat([index1,index2],dim=0).cuda() 
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0,max_node+1,1).repeat_interleave(2).view(1,-1),
                            torch.arange(max_edge+1,max_edge+1+max_node+1,1).repeat_interleave(2).view(1,-1)],dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1,1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def reverse_features(self, dia_len, features):
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features


    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_,2))
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll,aa,vv],dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll,aa,vv],dim=0)
                features =  torch.cat([features,features_temp],dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T,torch.LongTensor(tmp).T],1).cuda()

        return edge_index, features


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=4, score_function='scaled_dot_product', dropout=0.6):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class TextCNN(nn.Module):
    
    def __init__(self, num_classes):
        super(TextCNN, self).__init__()
        self.textf_input = nn.Conv1d(1024, 512, kernel_size=1, padding=0, bias=True)
        self.acouf_input = nn.Conv1d(300, 512, kernel_size=1, padding=0, bias=True)
        self.visuf_input = nn.Conv1d(342, 512, kernel_size=1, padding=0, bias=True)
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.linear4= nn.Linear(1536, 256)
        self.num_classes = num_classes

    def forward(self, textf, acouf, visuf, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        textf = (r1 + r2 + r3 + r4)/4
        textf = self.textf_input(textf.permute(1, 2, 0)).permute(2, 0, 1)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).permute(2, 0, 1)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).permute(2, 0, 1)
        textf = F.gelu(self.linear1(textf))
        acouf = F.gelu(self.linear1(acouf))
        visuf = F.gelu(self.linear1(visuf))
        x = torch.cat((textf, acouf, visuf), dim=-1)
        # project the features to the labels
        x = F.gelu(self.linear4(x))
        x = self.linear3(x)
        log_prob = F.log_softmax(x, 2)
        return log_prob
    
class LSTMModel(nn.Module):
    
    def __init__(self, D_m, D_h, n_classes=7, dropout=0.5):

        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=1666, hidden_size=256, bidirectional=True, dropout=0.5, num_layers=4)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=256, bidirectional=True, dropout=0.5, num_layers=4)
        self.fc1 = nn.Linear(1582, 512)
        self.fc2 = nn.Linear(342, 512)

        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)


        self.att = Attention(D_m)

        self.linear = nn.Linear(512, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, textf, acouf, visuf, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        textf = (r1 + r2 + r3 + r4)/4
        # print(textf.size(), acouf.size(), visuf.size())
        emotion = torch.cat((textf, acouf, visuf), dim=-1)
        emotion, _ = self.lstm(emotion)
        hidden = F.gelu(emotion)
        hidden = F.gelu(self.linear(hidden))
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob
    
    
class SimpleAttention(nn.Module):
    
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha

class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False, D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m + D_p,D_g)
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell = nn.GRUCell(D_p,D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)


        self.attention = SimpleAttention(D_g)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)

        return g_,q_,e_,alpha

class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, n_classes, listener_state=False, D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        self.dialogue_cell = DialogueRNNCell(self.D_m, D_g, D_p, D_e, listener_state, D_a, dropout)
        self.fc4 = nn.Linear(256, 512)
        self.linear = nn.Linear(1666, 512)
        self.smax_fc = nn.Linear(512, n_classes)

    def forward(self, textf, acouf, visuf, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        textf = (r1 + r2 + r3 + r4)/4

        U = torch.cat([textf, acouf, visuf], dim=-1)

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(U.type()) # batch, party, D_p
        e_ = torch.zeros(0).type(U.type()) # batch, D_e
        e = e_

        alpha = []
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])
        e = self.fc4(e) + self.linear(U)
        log_prob = F.log_softmax(self.smax_fc(e), 2)
        return log_prob
    
class M3Net(nn.Module):
    
    def __init__(self, args, base_model, D_m, D_g, D_e, graph_hidden_size, 
                 n_classes=7, dropout=0.5, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=False, use_modal=False, num_L = 3, num_K = 4):
        
        super(M3Net, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        self.args = args
        self.hidden_dim_a = D_g
        self.hidden_dim_v = D_g
        self.hidden_dim_l = D_g


        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            #self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 



        if self.graph_type=='hyper':
            self.graph_model = HyperGCN(a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64, nhidden=graph_hidden_size, nclass=n_classes, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
            print("construct "+self.graph_type)
        elif self.graph_type=='None':
            if not self.multi_modal:
                self.graph_net = nn.Linear(2*D_e, n_classes)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = nn.Linear(2*D_e, graph_hidden_size)
                if 'v' in self.modals:
                    self.graph_net_v = nn.Linear(2*D_e, graph_hidden_size)
                if 'l' in self.modals:
                    self.graph_net_l = nn.Linear(2*D_e, graph_hidden_size)
            print("construct Bi-LSTM")
        else:
            print("There are no such kind of graph")

        if self.multi_modal:
            #self.gatedatt = MMGatedAttention(D_g + graph_hidden_size, graph_hidden_size, att_type='general')
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)
        
        self.linear_fea = nn.Linear((D_g+graph_hidden_size)*len(self.modals), graph_hidden_size)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, textf, U_a, U_v, qmask, umask, seq_lengths, perturb=None):

        #=============roberta features
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
            else:
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim_a)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_a = U_a + perturb
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim_a)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_v = U_v + perturb
                if 'l' in self.modals:
                    U = self.linear_l(U)
                    emotions_l, hidden_l = self.lstm_l(U)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim_a)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_l = emotions_l + perturb

        if not self.multi_modal:
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []
        if self.graph_type=='GCN3' or self.graph_type=='DeepGCN':
            if not self.multi_modal:
                log_prob = self.graph_net(features, seq_lengths, qmask)
            else:
                emotions_a = self.graph_net_a(features_a, seq_lengths, qmask) if 'a' in self.modals else []
                emotions_v = self.graph_net_v(features_v, seq_lengths, qmask) if 'v' in self.modals else []
                emotions_l = self.graph_net_l(features_l, seq_lengths, qmask) if 'l' in self.modals else []

                if self.att_type == 'concat_subsequently':                
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                #elif self.att_type == 'gated':
                #    emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_l, self.modals)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                emotions_feat = nn.ReLU()(emotions_feat)
                log_prob = F.log_softmax(self.hidfc(self.smax_fc(emotions_feat)), 1)
        elif self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask)
            emotions_feat = self.dropout_(emotions_feat)
            emotions_feat = nn.ReLU()(emotions_feat)
            features = self.linear_fea(emotions_feat)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        else:
            print("There are no such kind of graph")        
        return log_prob, features


class M3Net2(nn.Module):
    
    def __init__(self, args, base_model, D_m, D_g, D_e, graph_hidden_size, 
                 n_classes=7, dropout=0.5, avec=False, 
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=False, use_modal=False, num_L = 3, num_K = 4):
        
        super(M3Net2, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)

        self.args = args
        self.hidden_dim_a = D_g
        self.hidden_dim_v = D_g
        self.hidden_dim_l = D_g

        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.dataset = dataset

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            #self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 


        if self.graph_type=='hyper':
            self.graph_model = HyperPathNN(a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64, nhidden=graph_hidden_size, nclass=n_classes, 
                                        dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, modals=self.modals, use_speaker=self.use_speaker, use_modal=self.use_modal, num_L=num_L, num_K=num_K)
            print("construct "+self.graph_type)
        elif self.graph_type=='None':
            if not self.multi_modal:
                self.graph_net = nn.Linear(2*D_e, n_classes)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = nn.Linear(2*D_e, graph_hidden_size)
                if 'v' in self.modals:
                    self.graph_net_v = nn.Linear(2*D_e, graph_hidden_size)
                if 'l' in self.modals:
                    self.graph_net_l = nn.Linear(2*D_e, graph_hidden_size)
            print("construct Bi-LSTM")
        else:
            print("There are no such kind of graph")

        if self.multi_modal:
            #self.gatedatt = MMGatedAttention(D_g + graph_hidden_size, graph_hidden_size, att_type='general')
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)
        
        self.linear_fea = nn.Linear((D_g+graph_hidden_size)*len(self.modals), graph_hidden_size)


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, textf, U_a, U_v, qmask, umask, seq_lengths, perturb=None):

        #=============roberta features
        [r1,r2,r3,r4]=textf
        seq_len, _, feature_dim = r1.size()

        r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features
        if self.base_model == 'LSTM':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
                if perturb is None:
                    perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim)
                    perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                    perturb.requires_grad_()
                    emotions = emotions + perturb
            else:
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim_a)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_a = U_a + perturb
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim_v)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_v = U_v + perturb
                if 'l' in self.modals:
                    U = self.linear_l(U)
                    emotions_l, hidden_l = self.lstm_l(U)
                    if perturb is None:
                        perturb_shape = (U.shape[0], U.shape[1], self.hidden_dim)
                        perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.args.delta, self.args.delta).to(self.args.device)
                        perturb.requires_grad_()
                    emotions_l = emotions_l + perturb


        if not self.multi_modal:
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []
            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []
            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []
        if self.graph_type=='GCN3' or self.graph_type=='DeepGCN':
            if not self.multi_modal:
                log_prob = self.graph_net(features, seq_lengths, qmask)
            else:
                emotions_a = self.graph_net_a(features_a, seq_lengths, qmask) if 'a' in self.modals else []
                emotions_v = self.graph_net_v(features_v, seq_lengths, qmask) if 'v' in self.modals else []
                emotions_l = self.graph_net_l(features_l, seq_lengths, qmask) if 'l' in self.modals else []

                if self.att_type == 'concat_subsequently':                
                    emotions = []
                    if len(emotions_a) != 0:
                        emotions.append(emotions_a)
                    if len(emotions_v) != 0:
                        emotions.append(emotions_v)
                    if len(emotions_l) != 0:
                        emotions.append(emotions_l)
                    emotions_feat = torch.cat(emotions, dim=-1)
                #elif self.att_type == 'gated':
                #    emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_l, self.modals)
                else:
                    print("There is no such attention mechnism")

                emotions_feat = self.dropout_(emotions_feat)
                emotions_feat = nn.ReLU()(emotions_feat)
                log_prob = F.log_softmax(self.hidfc(self.smax_fc(emotions_feat)), 1)
        elif self.graph_type=='hyper':
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask)
            emotions_feat = self.dropout_(emotions_feat)
            emotions_feat = nn.ReLU()(emotions_feat)
            features = self.linear_fea(emotions_feat)
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        else:
            print("There are no such kind of graph")        
        return log_prob, features