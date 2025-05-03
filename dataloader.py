import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
import torch.nn as nn
import copy

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

class IEMOCAPDataset(Dataset):
    def __init__(self, noise_type, percent, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.roberta1,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.num_classes = 4
        self.noise_indx = []
        for key in self.keys:
            self.videoLabels[key] = np.array(self.videoLabels[key])
            data = self.videoLabels[key]
            indices = np.where((data == 0) | (data == 1) | (data == 2) | (data == 3))[0]
            self.videoLabels[key] = np.array(self.videoLabels[key][indices])
            self.roberta1[key] = np.array(self.roberta1[key])[indices]
            self.roberta2[key] = np.array(self.roberta2[key])[indices]
            self.roberta3[key] = np.array(self.roberta3[key])[indices]
            self.roberta4[key] = np.array(self.roberta4[key])[indices]
            self.videoAudio[key] = np.array(self.videoAudio[key])[indices]
            self.videoVisual[key] = np.array(self.videoVisual[key])[indices]
            self.videoSpeakers[key] = np.array(self.videoSpeakers[key])[indices]
        
        keys_with_empty_arrays = [k for k, v in self.videoLabels.items() if isinstance(v, np.ndarray) and v.size == 0]
        
        self.videoSpeakers = {k: v for k, v in self.videoSpeakers.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoLabels = {k: v for k, v in self.videoLabels.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta1 = {k: v for k, v in self.roberta1.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta2 = {k: v for k, v in self.roberta2.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta3 = {k: v for k, v in self.roberta3.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta4 = {k: v for k, v in self.roberta4.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoAudio = {k: v for k, v in self.videoAudio.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoVisual = {k: v for k, v in self.videoVisual.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        
        self.keys = [item for item in self.keys if item not in keys_with_empty_arrays]
        self.videoLabels_gt = copy.deepcopy(self.videoLabels)
        for key in self.keys:            
            if noise_type == "asymmetric":
                for i in range(self.num_classes):
                    indices = np.where(self.videoLabels[key] == i)[0]
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j < percent * len(indices):
                            self.noise_indx.append(idx)
                            if i == 3:
                                self.videoLabels[key][idx] = 0
                            elif i == 2:
                                self.videoLabels[key][idx] = 1
                            elif i == 1:
                                self.videoLabels[key][idx] = 2
                            elif i == 0:
                                self.videoLabels[key][idx] = 3
            else:
                break
        self.len = len(self.keys)
        

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid][:, :300])),\
               torch.FloatTensor(np.array([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]])),\
               torch.FloatTensor(np.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(np.array(self.videoLabels[vid])),\
               torch.LongTensor(np.array(self.videoLabels_gt[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<10 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):
    def __init__(self, noise_type, percent, path="./data/MELD/meld_multimodal_features.pkl", train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.roberta1, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))
        self.noise_indx = []
        self.num_classes = 4

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        for key in self.keys:
            self.videoLabels[key] = np.array(self.videoLabels[key])
            data = self.videoLabels[key]
            indices = np.where((data == 0) | (data == 3) | (data == 4) | (data == 6))[0]
            indices_0 = np.where(data == 0)[0]
            self.videoLabels[key][indices_0] = 2
            indices_3 = np.where(data == 3)[0]
            self.videoLabels[key][indices_3] = 1
            indices_4 = np.where(data == 4)[0]
            self.videoLabels[key][indices_4] = 0
            indices_6 = np.where(data == 6)[0]
            self.videoLabels[key][indices_6] = 3
            self.videoLabels[key] = np.array(self.videoLabels[key][indices])
            self.roberta1[key] = np.array(self.roberta1[key])[indices]
            self.roberta2[key] = np.array(self.roberta2[key])[indices]
            self.roberta3[key] = np.array(self.roberta3[key])[indices]
            self.roberta4[key] = np.array(self.roberta4[key])[indices]
            self.videoAudio[key] = np.array(self.videoAudio[key])[indices]
            self.videoVisual[key] = np.array(self.videoVisual[key])[indices]
            self.videoSpeakers[key] = np.array(self.videoSpeakers[key])[indices]
        
        keys_with_empty_arrays = [k for k, v in self.videoLabels.items() if isinstance(v, np.ndarray) and v.size == 0]
        
        self.videoSpeakers = {k: v for k, v in self.videoSpeakers.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoLabels = {k: v for k, v in self.videoLabels.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta1 = {k: v for k, v in self.roberta1.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta2 = {k: v for k, v in self.roberta2.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta3 = {k: v for k, v in self.roberta3.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.roberta4 = {k: v for k, v in self.roberta4.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoAudio = {k: v for k, v in self.videoAudio.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        self.videoVisual = {k: v for k, v in self.videoVisual.items() if not (isinstance(v, np.ndarray) and v.size == 0)}
        
        self.keys = [item for item in self.keys if item not in keys_with_empty_arrays]
        
        self.videoLabels_gt = copy.deepcopy(self.videoLabels)
        for key in self.keys:
            if noise_type == "asymmetric":
                for i in range(self.num_classes):
                    indices = np.where(self.videoLabels[key] == i)[0]
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j < percent * len(indices):
                            self.noise_indx.append(idx)
                            if i == 3:
                                self.videoLabels[key][idx] = 0
                            elif i == 2:
                                self.videoLabels[key][idx] = 1
                            elif i == 1:
                                self.videoLabels[key][idx] = 2
                            elif i == 0:
                                self.videoLabels[key][idx] = 3
            else:
                break

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])),\
               torch.FloatTensor(np.array(self.roberta2[vid])),\
               torch.FloatTensor(np.array(self.roberta3[vid])),\
               torch.FloatTensor(np.array(self.roberta4[vid])),\
               torch.FloatTensor(np.array(self.videoVisual[vid])),\
               torch.FloatTensor(np.array(self.videoAudio[vid][:, :300])),\
               torch.FloatTensor(np.array(self.videoSpeakers[vid][:, :2])),\
               torch.FloatTensor(np.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(np.array(self.videoLabels[vid])),\
               torch.LongTensor(np.array(self.videoLabels_gt[vid])),\
               vid  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<10 else dat[i].tolist() for i in dat]

    
