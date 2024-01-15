import torch
import torch.nn as nn
import torch.nn.functional as F


class EuciLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EuciLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, target_vec, pos_vec, neg_vec):
        '''
        target_vec shape = (batch_size, inner_dim)
        pos_vec shape = (batch_size, inner_dim)
        neg_vec shape = (batch_size, num_of_negative, inner_dim)
        '''

        #  shape = batch_size
        pos_distance = F.mse_loss(target_vec, pos_vec)
        #  shape = batch_size, num_of_negative
        neg_distance = F.mse_loss(target_vec, neg_vec)

        #  print(weights)
        loss = F.logsigmoid(pos_distance) - torch.sum(F.logsigmoid(neg_distance), dim=-1)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss

    #  def __call__(self, input_data):
    #      return self.forward(input_data)
