import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from copy import deepcopy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    with open(config['result_path'],'a') as f:
        f.write(str(config['exp_name']))
        f.write('\n')
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('{} dataset: {} (x{}), {}'.format(config['dataset'],
            dataset[0][0].shape, len(dataset), dataset.n_classes))

    n_way = 5

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    n_shot, n_query = config['shot'], 15
    utils.log('{}-way {}-shot'.format(n_way, n_shot))
    n_batch = 1000
    ep_per_batch = 1
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=8)

    # model
    if config.get('model') is None: #classifier baseline
        if config.get('load') is None:
            if config.get('method') is None:
                model = models.make('meta-baseline', encoder=None)
            else:
                model = models.make('meta-baseline', encoder=None, method=config['method'])
        else:
            model = models.load(torch.load(config['load']))

        if config.get('load_encoder') is not None:
            
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder = encoder
        
    elif 'kgsa' in config.get('model'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)

            
        
    else:
        raise ValueError('Not valid model')
            


    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()


    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}
    va_lst = []
    test_epochs = args.test_epochs
    np.random.seed(config['seed'])
    
    for epoch in range(1,test_epochs+1):
  
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            with torch.no_grad():
                logits, loss = model(x_shot, x_query)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item(), len(data))
                aves['va'].add(acc, len(data))
                va_lst.append(acc)
          

               
                
        record = 'test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), _[-1])
        print(record)
        with open(config['result_path'],'a') as f:
            f.write(record)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test-epochs', type=int, default=2)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)

    
    if config.get('test_batch'):
        test_batch = config['test_batch']
        for exp in test_batch:
            print(exp)
            exp_config = {}
            exp_config = deepcopy(test_batch[exp])
            exp_config['result_path'] = config['result_path']
            exp_config['exp_name'] = exp
            main(exp_config)
    else:
        config['shot'] = args.shot
        main(config)

