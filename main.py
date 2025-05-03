import argparse
import torch
import os
from pre_training import first_brunch, second_brunch, EM_training
import numpy as np ,random
seed = 2024
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pretraining(args):
    first_brunch(args)
    second_brunch(args)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=5, metavar='E', help='number of epochs')
    
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

    parser.add_argument('--EM_epochs', type=int, default=5)
    parser.add_argument('--m', type=int, default=3)
    
    parser.add_argument('--e_threshold', type=float, default=0.3)
    parser.add_argument('--m_threshold', type=float, default=0.3)
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--projection_size', type=int, default=512)

    args = parser.parse_args() 
    print(args) 

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything()
    pretraining(args)
    EM_training(args)
    