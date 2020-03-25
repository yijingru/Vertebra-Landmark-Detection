import torch
import torch.nn as nn
import os
import numpy as np
from models import spinal_net
import decoder
import loss
from dataset import BaseDataset

def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}


    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 'state_dict': state_dict}
        torch.save(data, path)

    def load_model(self, model, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return model

    def train_network(self, args):
        save_path = 'weights_'+args.dataset
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        if args.ngpus>0:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=args.down_ratio)
                 for x in ['train', 'val']}

        dsets_loader = {'train': torch.utils.data.DataLoader(dsets['train'],
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=args.num_workers,
                                                             pin_memory=True,
                                                             drop_last=True,
                                                             collate_fn=collater),

                        'val':torch.utils.data.DataLoader(dsets['val'],
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=1,
                                                          pin_memory=True,
                                                          collate_fn=collater)}


        print('Starting training...')
        train_loss = []
        val_loss = []
        for epoch in range(1, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)
            scheduler.step(epoch)

            epoch_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion)
            val_loss.append(epoch_loss)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')
            np.savetxt(os.path.join(save_path, 'val_loss.txt'), val_loss, fmt='%.6f')

            if epoch % 10 == 0 or epoch ==1:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)), epoch, self.model)

            if len(val_loss)>1:
                if val_loss[-1]<np.min(val_loss[:-1]):
                    self.save_model(os.path.join(save_path, 'model_last.pth'), epoch, self.model)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict in data_loader:
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

