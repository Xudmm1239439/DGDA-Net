from models import MaskedNLLLoss, M3Net, M3Net2
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os
import scipy.sparse as sp
import numpy as np,random
import math
from itertools import cycle
import torch_geometric.nn as gnn
from loss import elr_loss
import warnings
warnings.filterwarnings("ignore")
seed = 2024
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
def distance_matrix(source, target, threshold=1000000):

    m, k = source.shape
    n, _ = target.shape

    if m*n*k < threshold:
        source = source.unsqueeze(1)
        target = target.unsqueeze(0)
        result = torch.sum((source - target) ** 2, dim=-1) ** (0.5)
    else:
        result = torch.empty((m, n))
        if m < n:
            for i in range(m):
                result[i, :] = torch.sum((source[i] - target)**2,dim=-1)**(0.5)
        else:
            for j in range(n):
                result[:, j] = torch.sum((source - target[j])**2,dim=-1)**(0.5)
    return result

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def MNN_training(args, model, edge_index, feature, label, size):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(feature, edge_index)

        loss = loss_func(pred[:size], label[:size].long())
        loss.backward()
        optimizer.step()

class MNN_GNN(nn.Module):
    def __init__(self, args, num_features, num_classes, conv_type='GIN', pool_type='TopK', emb=True) -> None:
        super().__init__()
        self.args = args
        self.num_features = num_features
        # print(num_classes)
        self.num_classes = num_classes
        self.hidden_dim = 512
        self.pooling_ratio = 0.5
        self.conv_type = conv_type
        self.pool_type = pool_type
        # self.K = K

        # self.embedding = nn.Sequential(nn.Linear(self.num_features, self.hidden_dim), nn.ReLU()) if emb else None

        if self.conv_type == 'GCN':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'SAGE':
            self.conv1 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv2 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv3 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        elif conv_type == 'GIN':
            self.conv1 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv2 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv3 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
        elif conv_type == 'GMT':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError("Invalid conv_type: %s" % conv_type)


        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.bn4 = nn.BatchNorm1d(self.hidden_dim)

        # Define Linear Layers
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        # self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = nn.Linear(self.hidden_dim//2, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, edge_index):
        edge_attr = None
        feature = x
        if torch.cuda.is_available():
            feature = feature.to(self.args.device)
            edge_index = edge_index.to(self.args.device)


        x1 = self.dropout(self.relu(self.conv1(x, edge_index, edge_attr), negative_slope=0.1))
        x = self.relu(x1, negative_slope=0.1)
        x = self.dropout(self.relu(self.bn4(x), negative_slope=0.1))
        x = feature + 0.01 * x
        x = self.relu(self.linear1(x), negative_slope=0.1)

        x = self.linear3(x)

        return x


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, device='cpu', source_dim=0, target_dim=0):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    # batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * source_dim + [[0]] * target_dim)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, device='cpu'):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)] #512,2

    def forward(self, input_list):
        
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
       
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low) 

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y
  
def train_EM(args, model, train_loader, optimizer, loss_func, config, i):
    model.train()

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(config["source_dataset"], 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        collate_fn=config["source_dataset"].collate_fn,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(config["target_dataset"], 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        collate_fn=config["source_dataset"].collate_fn,
                                        drop_last=True)

    class_num = config['num_class']

    random_layer = RandomLayer([args.projection_size, class_num], config["loss"]["random_dim"], config['device'])
   
    
    ad_net = AdversarialNetwork(config["loss"]["random_dim"], 32).to(config['device'])

    total_correct = 0
    all_fc = None
    loss_params = config["loss"]
    seed_everything()
    
    for inputs_source, inputs_target, data in zip(cycle(dset_loaders["source"]), cycle(dset_loaders["target"]), train_loader):
        optimizer.zero_grad()
        source_textf1, source_textf2, source_textf3, source_textf4, source_visuf, source_acouf, source_qmask, source_umask, \
                                    source_label, source_label_gt = [d.cuda() for d in inputs_source[:-1]] if args.cuda else inputs_source[:-1]
        source_acouf = source_acouf[:, :, :300]

        source_lengths = [(source_umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(source_umask))]

        source_log_prob, source_features = model([source_textf1,source_textf2,source_textf3,source_textf4], source_acouf, source_visuf, source_qmask, source_umask, source_lengths)

        source_label = torch.cat([source_label[j][:source_lengths[j]] for j in range(len(source_label))])
        source_label_gt = torch.cat([source_label_gt[j][:source_lengths[j]] for j in range(len(source_label_gt))])

        target_textf1, target_textf2, target_textf3, target_textf4, target_visuf, target_acouf, target_qmask, target_umask, \
                                    target_label, target_label_gt = [d.cuda() for d in inputs_target[:-1]] if args.cuda else inputs_target[:-1]
        target_acouf = target_acouf[:, :, :300]

        target_lengths = [(target_umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(target_umask))]

        target_log_prob, target_features = model([target_textf1,target_textf2,target_textf3,target_textf4], target_acouf, target_visuf, target_qmask, target_umask, target_lengths)

        target_label = torch.cat([target_label[j][:target_lengths[j]] for j in range(len(target_label))])
        target_label_gt = torch.cat([target_label_gt[j][:target_lengths[j]] for j in range(len(target_label_gt))])

        textf1, textf2, textf3, textf4, visuf, acouf, qmask, umask, \
                                    label, label_gt = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        acouf = acouf[:, :, :300]

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, feature_all = model([textf1, textf2, textf3, textf4], acouf, visuf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_gt = torch.cat([label_gt[j][:lengths[j]] for j in range(len(label_gt))])

        source_y = source_label
        target_y = target_label
        y_all = label

        features = torch.cat((source_features, target_features), dim=0)
        outputs = torch.cat((source_log_prob, target_log_prob), dim=0)
        y =  torch.cat((source_y, target_y), dim=0)
        softmax_out = torch.nn.Softmax(dim=-1)(outputs)
        entropy = Entropy(softmax_out)

        transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(i),
                    random_layer,config['device'], source_features.shape[0], target_features.shape[0])
        
        classifier_loss = loss_func(log_prob, y_all.long())

        for _ in range(args.m - 1):
            transfer_loss.backward()

            source_log_prob, source_features = model([source_textf1,source_textf2,source_textf3,source_textf4], source_acouf, source_visuf, source_qmask, source_umask, source_lengths)
            target_log_prob, target_features = model([target_textf1,target_textf2,target_textf3,target_textf4], target_acouf, target_visuf, target_qmask, target_umask, target_lengths)
            log_prob, feature_all = model([textf1, textf2, textf3, textf4], acouf, visuf, qmask, umask, lengths)

            source_y = source_label
            target_y = target_label
            y_all = label

            features = torch.cat((source_features, target_features), dim=0)
            outputs = torch.cat((source_log_prob, target_log_prob), dim=0)
            y =  torch.cat((source_y, target_y), dim=0)

            softmax_out = torch.nn.Softmax(dim=1)(outputs)
            entropy = Entropy(softmax_out)#熵

            transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(i),
                    random_layer,config['device'], source_features.shape[0], target_features.shape[0])
            classifier_loss = loss_func(log_prob, y_all.long())
            transfer_loss /= args.m

        two_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        two_loss.backward()
        optimizer.step()

        total_correct += int((outputs.argmax(dim=-1) == y).sum())
  

def mnn(source_feature, target_feature, topk=5):
    d_s_t = -distance_matrix(source_feature, target_feature)

    t_s_topk_index = d_s_t.topk(topk, dim=-1).indices
    s_t_topk_index = d_s_t.T.topk(topk, dim=-1).indices

    
    t_s_adjacency = torch.zeros((source_feature.shape[0], target_feature.shape[0]))
    s_t_adjacency = torch.zeros((target_feature.shape[0], source_feature.shape[0]))
    
    for i in range(source_feature.shape[0]):
        t_s_adjacency[i, t_s_topk_index[i]] = 1
    for j in range(target_feature.shape[0]):
        s_t_adjacency[j, s_t_topk_index[j]] = 1

    mnn_adjacency = t_s_adjacency * s_t_adjacency.T
    total_feature = torch.cat((source_feature, target_feature), dim=0)
    total_adj = torch.zeros((total_feature.shape[0], total_feature.shape[0]))
    total_adj[:source_feature.shape[0], source_feature.shape[0]:] = mnn_adjacency
    total_adj[source_feature.shape[0]:, :source_feature.shape[0]] = mnn_adjacency.T
    total_adj = total_adj + torch.eye(total_adj.shape[0])

    adj = sp.coo_matrix(total_adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  
    adj = torch.LongTensor(indices)  
    return adj

def get_train_valid_sampler(trainset, valid=0):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_source_valid_loaders(args, source_dataset, val_dataset, batch_size=32,  num_workers=0, pin_memory=False):
    if source_dataset == "MELD": 
        sourceset = MELDDataset(noise_type=args.noise_type, percent=args.percent, path='./data/MELD/meld_multimodal_features.pkl')
    elif source_dataset == "IEMOCAP": 
        sourceset = IEMOCAPDataset(noise_type=args.noise_type, percent=args.percent, path="./data/IEMOCAP/iemocap_multimodal_features.pkl")
    if val_dataset == "MELD": 
        valset = MELDDataset(noise_type=args.noise_type, percent=args.percent, path='./data/MELD/meld_multimodal_features.pkl')
    elif val_dataset == "IEMOCAP": 
        valset = IEMOCAPDataset(noise_type=args.noise_type, percent=args.percent, path="./data/IEMOCAP/iemocap_multimodal_features.pkl")

    source_train_sampler, source_valid_sampler = get_train_valid_sampler(sourceset)
    val_train_sampler, val_valid_sampler = get_train_valid_sampler(valset)

    train_loader = DataLoader(sourceset,
                              batch_size=batch_size,
                              sampler=source_train_sampler,
                              collate_fn=sourceset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(valset,
                              batch_size=batch_size,
                              sampler=val_train_sampler,
                              collate_fn=valset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader, valid_loader, sourceset, valset


def train(args, model, train_loader, optimizer, loss_func, elr=None):
    model.train()

    total_loss = 0
    total_correct = []
    all_feature = None
    all_y = None
    all_pred = None
    seed_everything()
    for data in train_loader:

        optimizer.zero_grad()

        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label, label_gt = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        acouf = acouf[:, :, :300]

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, features = model([textf1,textf2,textf3,textf4], acouf, visuf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_gt = torch.cat([label_gt[j][:lengths[j]] for j in range(len(label_gt))])

        if all_feature == None:
            all_feature = features#
            all_y = label#
            all_pred = log_prob#
        else:
            all_feature = torch.cat((all_feature, features), dim=0)
            all_y = torch.cat((all_y, label), dim=0)
            all_pred = torch.cat((all_pred, log_prob), dim=0)

        loss = loss_func(log_prob, label)#loss_function

        if elr is not None:
            elr_loss = elr(log_prob, label)
            loss += elr_loss

        loss.backward()

        total_loss += float(loss)

        total_correct.append(int((log_prob.argmax(dim=-1) == label).sum()) / len(label))

        optimizer.step()

    return total_loss / len(train_loader), np.mean(total_correct), all_feature, all_y, all_pred

@torch.no_grad()
def test(args, model, loader):

    model.eval()

    total_correct = []
    all_feature = None
    all_y = None
    all_label = None
    all_pred = None
    seed_everything()
    for data in loader:

        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label, label_gt = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        acouf = acouf[:, :, :300]

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, features = model([textf1,textf2,textf3,textf4], acouf, visuf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_gt = torch.cat([label_gt[j][:lengths[j]] for j in range(len(label_gt))])

        if all_feature == None:
            all_feature = features
            all_y = log_prob.argmax(dim=-1)
            all_label = label
            all_pred = log_prob
        else:
            all_feature = torch.cat((all_feature, features), dim=0)
            all_y = torch.cat((all_y, log_prob.argmax(dim=-1)), dim=0)
            all_pred = torch.cat((all_pred, log_prob), dim=0)
            all_label = torch.cat((all_label, label), dim=0)

        total_correct.append(int((log_prob.argmax(dim=-1) == label).sum()) / len(label))

    return np.mean(total_correct), all_feature, all_y, all_pred, all_label


@torch.no_grad()
def inference(args, model, loader, loss_func=nn.NLLLoss()):

    model.eval()

    total_correct = []
    all_feature = None
    all_y = None
    all_label = None
    all_pred = None
    total_loss = 0
    seed_everything()
    for data in loader:

        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label, label_gt = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        acouf = acouf[:, :, :300]

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, features = model([textf1,textf2,textf3,textf4], acouf, visuf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_gt = torch.cat([label_gt[j][:lengths[j]] for j in range(len(label_gt))])

        if all_feature == None:
            all_feature = features
            all_y = log_prob.argmax(dim=-1)
            all_label = label
            all_pred = log_prob
        else:
            all_feature = torch.cat((all_feature, features), dim=0)
            all_y = torch.cat((all_y, log_prob.argmax(dim=-1)), dim=0)
            all_pred = torch.cat((all_pred, log_prob), dim=0)
            all_label = torch.cat((all_label, label), dim=0)

        loss = loss_func(log_prob, label)
        total_loss += float(loss)

        total_correct.append(int((log_prob.argmax(dim=-1) == label).sum()) / len(label))

    return all_feature, all_pred, all_label, np.mean(total_correct), total_loss / len(loader)

def first_brunch(args):

    print('Pre-training first brunch')
    args.method = 'first'

    feat2dim = {'denseface':342,'audio':300}
    D_audio = feat2dim['audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = 1024
    D_g = 512
    D_e = 256
    graph_h = 512
    n_classes  = 4

    train_loader, valid_loader, train_dataset, valid_dataset = get_source_valid_loaders(args, args.source_dataset, 
                                                          args.val_dataset, 
                                                          batch_size=32,  
                                                          num_workers=0, 
                                                          pin_memory=False)
    seed_everything()
    model = M3Net(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
                no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
                graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
                D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
                use_modal=args.use_modal, num_L = 3, num_K = 4).to(args.device)
    

    loss_function  = nn.NLLLoss()

    elr = elr_loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_val = 0

    for epoch in tqdm(range(1, args.epochs+1)):

        train_loss, train_acc, feature, label, fc = train(args, model, train_loader, optimizer, loss_function, elr)

        test_acc, test_feature, test_label, _, _ = test(args, model, valid_loader)

        if test_acc > best_val:
            best_val = test_acc
            best_gmt_train_feature = feature
            best_gmt_target_feature = test_feature
            torch.save(model.state_dict(),
                       os.path.join("./", f'pretraining/first_{args.source_dataset}_{args.val_dataset}.pth'))
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, train acc: {train_acc}'
                  f'Test Acc: {test_acc:.4f}')
        
    return best_gmt_train_feature, best_gmt_target_feature, label, test_label

def second_brunch(args):
    print('Pre-training second brunch')
    args.method = 'second'

    feat2dim = {'denseface':342,'audio':300}
    D_audio = feat2dim['audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = 1024
    D_g = 512
    D_p = 150
    D_e = 256
    D_h = 100
    D_a = 100
    graph_h = 512
    n_classes  = 4

    train_loader, valid_loader, train_dataset, valid_dataset = get_source_valid_loaders(args, args.source_dataset, 
                                                          args.val_dataset, 
                                                          batch_size=32,  
                                                          num_workers=0, 
                                                          pin_memory=False)
    seed_everything()
    model = M3Net2(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
            no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
            graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
            D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
            use_modal=args.use_modal, num_L = 3, num_K = 4).to(args.device)
    
    loss_function  = nn.NLLLoss()

    elr = elr_loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_val = 0

    for epoch in tqdm(range(1, args.epochs+1)):

        train_loss, train_acc, feature, label, fc = train(args, model, train_loader, optimizer, loss_function, elr)

        test_acc, test_feature, test_label, _, _ = test(args, model, valid_loader)

        if test_acc > best_val:
            best_val = test_acc
            best_gmt_train_feature = feature
            best_gmt_target_feature = test_feature
            torch.save(model.state_dict(),
                       os.path.join("./", f'pretraining/second_{args.source_dataset}_{args.val_dataset}.pth'))
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, train acc: {train_acc}'
                  f'Test Acc: {test_acc:.4f}')
        
    return best_gmt_train_feature, best_gmt_target_feature, label, test_label

def EM_training(args):

    print("Doupling dual branches")

    feat2dim = {'denseface':342,'audio':300}
    D_audio = feat2dim['audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = 1024
    D_g = 512
    D_p = 150
    D_e = 256
    D_h = 100
    D_a = 100
    graph_h = 512
    n_classes  = 4

    e_step_label_acc = []
    m_step_label_acc = []
    args.method = 'first'

    source_first_loader, target_first_loader, source_first_dataset, target_first_dataset = get_source_valid_loaders(args, args.source_dataset, 
                                                        args.val_dataset, 
                                                        batch_size=32,  
                                                        num_workers=0, 
                                                        pin_memory=False)
    seed_everything()
    model_first = M3Net(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
                no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
                graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
                D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
                use_modal=args.use_modal, num_L = 3, num_K = 4).to(args.device)

    model_first.load_state_dict(torch.load(f'pretraining/first_{args.source_dataset}_{args.val_dataset}.pth'))
    optimizer_first = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_function  = nn.NLLLoss()

    target_acc, _, _, _, _ = test(args, model_first, target_first_loader)          
    print('Direct predict first acc:', target_acc)

    args.method = 'second'
    source_second_loader, target_second_loader, source_second_dataset, target_second_dataset = get_source_valid_loaders(args, args.source_dataset, 
                                                    args.val_dataset, 
                                                    batch_size=32,  
                                                    num_workers=0, 
                                                    pin_memory=False)
    seed_everything()
    model_second = M3Net2(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
            no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
            graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
            D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
            use_modal=args.use_modal, num_L = 3, num_K = 4).to(args.device)

    model_second.load_state_dict(torch.load(f'pretraining/second_{args.source_dataset}_{args.val_dataset}.pth'))
    optimizer_second = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)

    target_acc, _, _, _, _ = test(args, model_second, target_second_loader)                
    print('Direct predict second acc:', target_acc)


    config = {}
    config["loss"] = {"trade_off": 1.0}
    config['num_class'] = 4
    
    if torch.cuda.is_available():
        config['device'] = args.device
    else:
        config['device'] = 'cpu'
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    total_best_train_loss = 10000

    ################## coupling 
    top_E, top_M = 0,0  
    seed_everything()
    for em_step in range(args.EM_epochs):
        if os.path.exists(f'pretraining/M_second_{args.source_dataset}_{args.val_dataset}.pth'):
            model_second.load_state_dict(torch.load(f'pretraining/M_second_{args.source_dataset}_{args.val_dataset}.pth'))
        source_second_feature, source_second_pred, source_second_label, _, _ = inference(args, model_second, source_second_loader)
        target_second_feature, target_second_pred, target_second_label, _, _ = inference(args, model_second, target_second_loader)
        target_second_pred = torch.nn.Softmax(-1)(target_second_pred)
        '''
        统计label的占比
        '''
        pesudo_second_idx = torch.where(target_second_pred > args.e_threshold)[0]
        size_pesudo_second = pesudo_second_idx.shape[0]
        
        if size_pesudo_second <= 0:
            pesudo_second_idx = torch.arange(1).long().to(args.device)
            pesudo_second_label = target_second_pred[pesudo_second_idx].argmax(dim=-1)
        else:
            pesudo_second_label = target_second_pred[pesudo_second_idx].argmax(dim=-1)
            ture_second_label = target_second_label[pesudo_second_idx]
            e_acc = (pesudo_second_label==ture_second_label).sum()/len(pesudo_second_idx)
            print('e_acc:', (pesudo_second_label==ture_second_label).sum()/len(pesudo_second_idx))
            e_step_label_acc.append(e_acc.item())

        edge_index = mnn(source_second_feature, target_second_feature)

        pesudo_second_idx = [num % args.batch_size for num in pesudo_second_idx]

        source_first, target_first_copy = source_second_dataset, target_second_dataset
        
        source_first_loader = source_second_loader
        target_second_dataloader = DataLoader(
            Subset(target_first_copy, pesudo_second_idx),
            batch_size=args.batch_size,
            collate_fn=target_first_copy.collate_fn
        )

        E_training_data = source_first + Subset(target_first_copy, pesudo_second_idx)
        
        E_train_loader = DataLoader(E_training_data, 
                                    batch_size=args.batch_size,
                                    collate_fn=target_first_copy.collate_fn)
        
        config["source_dataset"] = source_first
        config["target_dataset"] = Subset(target_first_copy, pesudo_second_idx)

        train_EM(args, model_first, E_train_loader, optimizer_first, loss_function, config, em_step)

        E_source_first_feature, E_source_pred, E_source_first_label, E_source_acc, E_source_loss = inference(args, model_first, source_first_loader)
        E_target_first_feature, E_target_pred, E_target_first_label, E_target_acc, _ = inference(args, model_first, target_first_loader)

        if total_best_train_loss > E_source_loss:
            best_E_source_first_feature = E_source_first_feature
            best_E_target_first_feature = E_target_first_feature
            print(f'Epoch: {em_step}, E step train acc: {E_source_acc}, target acc: {E_target_acc.item()}')
            torch.save(model_first.state_dict(),
                        os.path.join("./", f'pretraining/E_first_{args.source_dataset}_{args.val_dataset}_E.pth'))
            
        if top_E < E_target_acc.item():
            top_E = E_target_acc.item()

        mnn_model = MNN_GNN(args, num_features=D_g, num_classes=4,
                conv_type='GCN', pool_type=args.pool_type).to(args.device)

        E_feature = torch.cat((best_E_source_first_feature, best_E_target_first_feature), dim=0)
        E_label = torch.cat((E_source_first_label, E_target_first_label), dim=0)

        MNN_training(args, mnn_model, edge_index, E_feature, E_label, best_E_source_first_feature.shape[0])

        E_pred = mnn_model(E_feature, edge_index)
        E_target_pred = E_pred[E_source_first_feature.shape[0]:]
        E_target_pred = torch.nn.Softmax(-1)(E_target_pred)

        pesudo_first_idx = torch.where(E_target_pred > args.m_threshold)[0]
        size_pesudo_first = pesudo_first_idx.shape[0]       
        if size_pesudo_first <= 0:
            pesudo_first_idx = torch.arange(1).long().to(args.device)
            pesudo_first_label = E_target_pred[pesudo_first_idx].argmax(dim=-1)
        else:
            pesudo_first_label = E_target_pred[pesudo_first_idx].argmax(dim=-1)
            ture_first_label = E_target_first_label[pesudo_first_idx]
            m_acc = (pesudo_first_label == ture_first_label).sum() / len(pesudo_first_idx)
            print('m_acc:', (pesudo_first_label == ture_first_label).sum() / len(pesudo_first_idx))
            m_step_label_acc.append(m_acc.item())

        pesudo_first_idx = [num % args.batch_size for num in pesudo_first_idx]

        source_second = source_second_dataset
        target_second_copy = target_second_dataset

        M_training_data = source_second + Subset(target_second_copy, pesudo_first_idx)

        M_train_loader = DataLoader(M_training_data, 
                                batch_size=args.batch_size,
                                collate_fn=target_second_copy.collate_fn)

        config["source_dataset"] = source_second
        config["target_dataset"] = Subset(target_second_copy, pesudo_first_idx)

        train_EM(args, model_second, M_train_loader, optimizer_second, loss_function, config, em_step)
        M_source_second_feature, M_source_pred, M_source_second_label, M_source_acc, M_source_loss = inference(args, model_second,
                                                                                                source_second_loader)
        M_target_second_feature, M_target_pred, M_target_second_label, M_target_acc, _ = inference(args, model_second,
                                                                                        target_second_loader)
        if total_best_train_loss > M_source_loss:

            print(f'Epoch: {em_step}, M step train acc: {M_source_acc}, target acc: {M_target_acc.item()}')
            torch.save(model_second.state_dict(),
                        os.path.join("./", f'pretraining/M_second_{args.source_dataset}_{args.val_dataset}_M.pth'))
    
        if top_M < M_target_acc.item():
            top_M =  M_target_acc.item()

    e_step_label_acc = [round(num, 3) for num in e_step_label_acc]
    m_step_label_acc = [round(num, 3) for num in m_step_label_acc]
    print(f'e_step_select_pseudo_label_acc:{e_step_label_acc}')
    print(f'm_step_select_pseudo_label_acc:{m_step_label_acc}')
    print(f"top_E, top_M: {top_E}, {top_M}")





    







