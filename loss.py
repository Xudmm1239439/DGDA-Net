import torch.nn as nn
import torch
import torch.nn.functional as F

class elr_loss(nn.Module):
    def __init__(self, num_classes=4, beta=0.3):
        '''Early Learning Regularization.
         Parameters
         * `num_examp` Total number of training examples.
         * `num_classes` Number of classes in the classification problem.
         * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
         * `beta` Temporal ensembling momentum for target estimation.
         '''
        
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.beta = beta
        self.lambd = 3
        

    def forward(self, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output,dim=1)#输出转换为概率分布
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)#限制数值范围
        y_pred_ = y_pred.data.detach()#断开梯度计算
        target = self.beta + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(target * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.lambd * elr_reg
        return  final_loss