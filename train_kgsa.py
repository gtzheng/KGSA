import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler

import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h
class SaverPool:
    def __init__(self, pool_size, save_path):
        self.pool_size = pool_size
        self.min_val = 0.0
        self.min_pos = 0
        self.val_arr = np.zeros(pool_size)
        self.save_path = save_path
    def save(self, va, save_obj):
        if self.min_val < va:
            self.val_arr[self.min_pos] = va
            torch.save(save_obj, os.path.join(self.save_path, 'epoch-{}.pth'.format(self.min_pos)))
            self.min_pos = np.argmin(self.val_arr)
            self.min_val = self.val_arr[self.min_pos]

def main(config):
    svname = args.name
    if svname is None:
        svname = '_{}_{}-{}shot'.format(config['model'],
                config['train_dataset'], config['n_shot'])
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join(config['save_path'], svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(config['tensorboard_path'], svname))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))

    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot+n_query,
            ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=config['num_workers'], pin_memory=False)
    batch_loader = DataLoader(train_dataset, 128, shuffle=False,
                              num_workers=8, pin_memory=False)
    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        tval_sampler = CategoriesSampler(
                tval_dataset.label, 1000,
                n_way, n_shot + n_query,
                ep_per_batch=1)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))

    val_sampler = CategoriesSampler(
            val_dataset.label, 1000,
            n_way, n_shot + n_query,
            ep_per_batch=1)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
        if config.get('load_encoder'):
            if 'ssl_T' in config.get('load_encoder'):
                model_sv = torch.load(config['load_encoder'])
                model.encoder.load_state_dict(model_sv['model'],strict=False)
            else:
                model_sv = torch.load(config['load_encoder'])
                model_sv['model_args']['encoder'] = 'resnet12'
                loaded_model = models.load(model_sv)
                encoder = loaded_model.encoder
                model.encoder.load_state_dict(encoder.state_dict())
            
                
    if config.get('_parallel'):
        model = nn.DataParallel(model)

    save_epoch = config.get('save_epoch')
    max_va = 0.
    max_tva = 0.0
    max_ta = 0.0
    tol_count = 0
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    max_epoch = config['max_epoch']
    saver = SaverPool(5,save_path)
    for epoch in range(1, max_epoch + 1):
        model.train()
        
        aves = {k: utils.Averager() for k in aves_keys}
        np.random.seed(config['seed']+epoch)
        with tqdm(train_loader, total=len(train_loader), desc='train', leave=False) as pbar:
            for data, ori_label in pbar:
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_train_way, n_train_shot, n_query,
                        ep_per_batch=ep_per_batch)
                label = fs.make_nk_label(n_train_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()
                logits, loss = model(x_shot, x_query)
                acc = utils.compute_acc(logits, label)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                aves['tl'].add(loss.item())
                aves['ta'].add(acc)
                logits = None; loss = None 
                pbar.set_postfix({"Acc":'{0:.4f}'.format(aves['ta'].item()), 
                                    'Loss':'{0:.4f}'.format(aves['tl'].item()),
                                    "Temp":'{0:.4f}'.format(model.temp.item())
                                })
    
        #eval
        model.eval()
        tva_lst = []
   
        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            with tqdm(loader, total=len(loader), desc=name, leave=False) as pbar:
                for idx, (data, ori_label) in enumerate(pbar):
                    x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_shot, n_query,
                            ep_per_batch=1)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=1).cuda()

                    with torch.no_grad():
                        logits, _ = model(x_shot, x_query)
                        loss = F.cross_entropy(logits, label)
                        acc = utils.compute_acc(logits, label)
                        aves[name_l].add(loss.item())
                        aves[name_a].add(acc)
                        if name == 'tval':
                            tva_lst.append(acc)

                    pbar.set_postfix({"Acc":'{0:.4f}'.format(aves[name_a].item()), 
                                        'Loss':'{0:.4f}'.format(aves[name_l].item())
                                    })
                    


        # post
        if lr_scheduler is not None:
            lr_scheduler.step()


        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
 
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}, tval {:.4f}, '
                'val {:.4f}, loss {:.4f}, t {:.4f} {}/{}'.format(
                epoch, aves['ta'], aves['tva'],
                aves['va'], aves['tvl'], model.temp.item(), t_used, t_estimate))

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),
            'training': training
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        saver.save(aves['va'],save_obj)
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
                    
        writer.add_scalar('loss/train', aves['tl'], epoch)
        writer.add_scalar('loss/tval', aves['tvl'], epoch)
        writer.add_scalar('loss/val', aves['vl'], epoch)

        writer.add_scalar('acc/train', aves['ta'], epoch)
        writer.add_scalar('acc/tval', aves['tva'], epoch)
        writer.add_scalar('acc/val', aves['va'], epoch)

        writer.flush()
        if aves['va'] > max_va:
            max_va = aves['va']
            max_tva = aves['tva']
            max_ta = aves['ta']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
            tol_count = 0
        else:
            tol_count += 1
            if tol_count == config['tolerance']:
                break # early stopping
    utils.log('[ES], train: {:.4f}, val: {:.4f}, test: {:.4f}, (+-{:.4f})'.format(max_ta, max_va, max_tva, 0.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)


    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.tag == 'test':
        config['num_workers'] = 0
    else:
        config['num_workers'] = 8
    main(config) 

