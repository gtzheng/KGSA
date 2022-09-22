import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        print('Meta-Baseline, method {} temp {}'.format(method, temp))

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        elif self.method == 'cl2n':
            avg_fea = torch.cat((x_shot.reshape(shot_shape[0],shot_shape[1]*shot_shape[2],-1), x_query), dim=1).mean(dim=1,keepdim=True)
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot-avg_fea, dim=-1)
            x_query = F.normalize(x_query-avg_fea, dim=-1)
            metric = 'cl2n'
        elif self.method == 'knn':
            return x_shot, x_query
        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits

