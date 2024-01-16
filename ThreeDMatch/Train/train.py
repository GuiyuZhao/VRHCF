import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import shutil
import sys
import argparse
import torch 

sys.path.append('../../')
from ThreeDMatch.Train.dataloader import get_dataloader
from ThreeDMatch.Train.trainer import Trainer
from module.SphereNet import SphereNet
from torch import optim


class Args(object):
    def __init__(self):
        self.experiment_id = "Proposal" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        shutil.copy2(os.path.join('', 'trainer.py'), os.path.join(snapshot_root, 'trainer.py'))
        shutil.copy2(os.path.join('', '../../module/SphereNet.py'), os.path.join(snapshot_root, 'SphereNet.py'))
        shutil.copy2(os.path.join('', '../../module/SphericalCNN.py'), os.path.join(snapshot_root, 'SphericalCNN.py'))
        shutil.copy2(os.path.join('', '../../loss/desc_loss.py'), os.path.join(snapshot_root, 'loss.py'))
        self.epoch = 40
        self.batch_size = 10
        self.rad_n = 15
        self.azi_n = 40
        self.ele_n = 20
        self.des_r = 0.3
        
        self.dataset = '3DMatch'
        self.data_train_dir = '../../data/3DMatch/patches'
        self.data_val_dir = '../../data/3DMatch/patches'

        self.gpu_mode = True
        self.verbose = True
        self.freeze_epoch = 5

        # model & optimizer
        self.model = SphereNet(self.des_r, self.rad_n, self.azi_n, self.ele_n, self.dataset)
        self.pretrain = ''
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 5

        # dataloader
        self.train_loader = get_dataloader(root=self.data_train_dir,
                                           batch_size=self.batch_size,
                                           split='train1',
                                           shuffle=True,
                                           num_workers=0,  # if the dataset is offline generated, must 0
                                           )
        self.val_loader = get_dataloader(root=self.data_val_dir,
                                         batch_size=self.batch_size,
                                         split='val1',
                                         shuffle=False,
                                         num_workers=0,  # if the dataset is offline generated, must 0
                                         )

        print("Training set size:", self.train_loader.dataset.__len__())
        print("Validate set size:", self.val_loader.dataset.__len__())

        # snapshot
        self.snapshot_interval = int(self.train_loader.dataset.__len__() / self.batch_size / 2)
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        # evaluate
        self.evaluate_interval = 1

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':

    args = Args()
    trainer = Trainer(args)
    trainer.train()
