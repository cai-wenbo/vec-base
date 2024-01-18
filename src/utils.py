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
        neg_vec shape = (batch_size, inner_dim)
        '''

        #  shape = batch_size

        #  pos_score = torch.square(torch.cdist(target_vec.unsqueeze(dim = 1), pos_vec.unsqueeze(dim = 1)).squeeze(dim = 1).squeeze(dim = 1))
        #  neg_score_1 = torch.square(torch.cdist(target_vec.unsqueeze(dim = 1), neg_vec_1.unsqueeze(dim = 1)).squeeze(dim = 1).squeeze(dim = 1))
        #  neg_score_2 = torch.square(torch.cdist(target_vec.unsqueeze(dim = 1), neg_vec_2.unsqueeze(dim = 1)).squeeze(dim = 1).squeeze(dim = 1))


        #  pos_score = torch.matmul(target_vec.unsqueeze(1), pos_vec.unsqueeze(-1)).squeeze(dim=1).squeeze(dim = 1)
        #  neg_score_1 = torch.matmul(target_vec.unsqueeze(1), neg_vec_1.unsqueeze(-1)).squeeze(dim=1).squeeze(dim = 1)
        #  neg_score_2 = torch.matmul(target_vec.unsqueeze(1), neg_vec_2.unsqueeze(-1)).squeeze(dim=1).squeeze(dim = 1)


        pos_score = F.cosine_similarity(target_vec, pos_vec)
        neg_score = F.cosine_similarity(target_vec.unsqueeze(1), neg_vec, dim = 2)

        #  shape = batch_size, num_of_negative
        #  neg_score = F.cosine_similarity(target_vec, neg_vec)

        #  print(weights)
        loss =  -(F.logsigmoid(pos_score) + torch.sum(F.logsigmoid(-neg_score), dim = -1)/ neg_score.shape[1])
        #  loss  = pos_score - neg_score
        #  loss = pos_distance - neg_distance
        #  loss =  pos_score - 1 / 2  * (neg_score_1 + neg_score_2)

        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        #  print(loss)
        return loss

    #  def __call__(self, input_data):
    #      return self.forward(input_data)
