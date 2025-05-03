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
import numpy as np ,random
import math
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import argparse
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


def eval_graph_model(model, loss_function, dataloader, cuda, optimizer=None):
    losses, preds, labels, labels_gt = [], [], [], []
    masks = []
    vids = []

    model.eval()
    seed_everything()
    for data in dataloader:
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label, label_gt = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        acouf = acouf[:, :, :300]

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, _ = model([textf1,textf2,textf3,textf4], acouf, visuf, qmask, umask, lengths)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        label_gt = torch.cat([label_gt[j][:lengths[j]] for j in range(len(label_gt))])
        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        labels_gt.append(label_gt.cpu().numpy())
        losses.append(loss.item())
        masks.append(umask.view(-1).cpu().numpy())
            

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        labels_gt = np.concatenate(labels_gt)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    vids += data[-1]
    labels = np.array(labels)
    labels_gt = np.array(labels_gt)
    preds = np.array(preds)
    vids = np.array(vids)
    masks = np.array(masks)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, labels_gt, preds, avg_fscore, masks


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--graph_type', default='hyper', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--graph_construct', default='direct', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--source_dataset', default='MELD', help='dataset to train and test')
    
    parser.add_argument('--val_dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--pool_type', type=str, choices=['TopK', 'Edge', 'SAG', 'ASA','GMT'], default='TopK')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='BN', help='NORM type')

    parser.add_argument('--conv_type', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN'], default='GCN')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--testing', action='store_true', default=False, help='testing')
    
    parser.add_argument('--noise_type', type=str, default='asymmetric', help="Noise Label Type (asymmetric or no)") 

    parser.add_argument('--percent', type=float, default=0.4, help="Noise Labels Ratio, Asymmetric [0.1, 0.2, 0.3, 0.4]") 

    parser.add_argument('--delta', type=float, default=8e-1)

    parser.add_argument('--EM_epochs', type=int, default=1)
    parser.add_argument('--m', type=int, default=3)
    
    parser.add_argument('--e_threshold', type=float, default=0.3)
    parser.add_argument('--m_threshold', type=float, default=0.3)
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--projection_size', type=int, default=512)

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
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
    seed_everything()
    model = M3Net(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
                no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
                graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
                D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
                use_modal=args.use_modal, num_L = 3, num_K = 4)
    
    model2 = M3Net2(args, args.base_model, D_m, D_g, D_e, graph_h, n_classes=n_classes, dropout=args.dropout,
            no_cuda=args.no_cuda, graph_type=args.graph_type, use_topic=False, alpha=0.2, multiheads=2,
            graph_construct=args.graph_construct, use_GCN=False, use_residue=args.use_residue, D_m_v = D_visual,
            D_m_a = D_audio, modals=args.modals, att_type="concat_DHT", av_using_lstm=False,
            use_modal=args.use_modal, num_L = 3, num_K = 4)
    if cuda:
        model.cuda()
        model2.cuda()

    train_loader, valid_loader, _, _ = get_source_valid_loaders(args, args.source_dataset, args.val_dataset, batch_size=32,  num_workers=0, pin_memory=False)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    ### first ###
    model.load_state_dict(torch.load(f'./pretraining/first_{args.source_dataset}_{args.val_dataset}.pth'))
    test_loss, test_acc, test_label, test_label_gt, test_pred, test_fscore, test_mask = eval_graph_model(model, loss_function, valid_loader, cuda)

    all_fscore.append(test_fscore)

    if best_loss == None or best_loss > test_loss:
        best_loss, best_label, best_pred = test_loss, test_label_gt, test_pred
        best_mask = test_mask

    if best_fscore == None or best_fscore < test_fscore:
        best_fscore = test_fscore
        best_label, best_pred = test_label_gt, test_pred

    best_mask = None
    print("---------------first---------------------")
    print ('----------best F-Score:', max(all_fscore))
    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    ### second ###
    model2.load_state_dict(torch.load(f'./pretraining/second_{args.source_dataset}_{args.val_dataset}.pth'))
    test_loss, test_acc, test_label, test_label_gt, test_pred, test_fscore, test_mask = eval_graph_model(model2, loss_function, valid_loader, cuda)

    all_fscore.append(test_fscore)

    if best_loss == None or best_loss > test_loss:
        best_loss, best_label, best_pred = test_loss, test_label_gt, test_pred
        best_mask = test_mask

    if best_fscore == None or best_fscore < test_fscore:
        best_fscore = test_fscore
        best_label, best_pred = test_label_gt, test_pred

    best_mask = None
    print("---------------second---------------------")
    print ('----------best F-Score:', max(all_fscore))
    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

    ### E first ###
    model.load_state_dict(torch.load(f'pretraining/E_first_{args.source_dataset}_{args.val_dataset}_E.pth'))
    test_loss, test_acc, test_label, test_label_gt, test_pred, test_fscore, test_mask = eval_graph_model(model, loss_function, valid_loader, cuda)

    all_fscore.append(test_fscore)

    if best_loss == None or best_loss > test_loss:
        best_loss, best_label, best_pred = test_loss, test_label_gt, test_pred
        best_mask = test_mask

    if best_fscore == None or best_fscore < test_fscore:
        best_fscore = test_fscore
        best_label, best_pred = test_label_gt, test_pred

    best_mask = None
    print("---------------first---------------------")
    print ('----------best F-Score:', max(all_fscore))
    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

    ### M second ###
    model2.load_state_dict(torch.load(f'pretraining/M_second_{args.source_dataset}_{args.val_dataset}_M.pth'))
    test_loss, test_acc, test_label, test_label_gt, test_pred, test_fscore, test_mask = eval_graph_model(model, loss_function, valid_loader, cuda)

    all_fscore.append(test_fscore)

    if best_loss == None or best_loss > test_loss:
        best_loss, best_label, best_pred = test_loss, test_label_gt, test_pred
        best_mask = test_mask

    if best_fscore == None or best_fscore < test_fscore:
        best_fscore = test_fscore
        best_label, best_pred = test_label_gt, test_pred

    best_mask = None
    print("---------------first---------------------")
    print ('----------best F-Score:', max(all_fscore))
    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

