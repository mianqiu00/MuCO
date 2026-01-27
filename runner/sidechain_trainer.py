import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import math
import yaml

import lightning as L
from lightning.fabric import Fabric

from model.sidechain.utils.loader import load_seed, load_ema, load_config
from model.sidechain.utils.logger import Logger, set_log, start_log
from model.sidechain.utils.train_utils import count_parameters
from model.sidechain.dataset_cluster import get_dataloader
from model.sidechain.utils.structure_utils import create_structure_from_crds
from model.sidechain.loss import CFMLoss
from model.sidechain.models.cnf import CNF
from model.sidechain.models.equiformer_v2.equiformer_v2 import EquiformerV2
from model.sidechain.utils.metrics import metrics_per_chi, atom_rmsd

class SidechainTrainer(object):
    def __init__(self, config, fabric: Fabric, timestamp: str):
        super(SidechainTrainer, self).__init__()
        self.config = config
        self.fabric = fabric
        self.timestamp = timestamp
        
        # 1. 定义并创建目录结构 (仅限 Rank 0)
        self.run_dir = Path("saved_model") / self.timestamp
        self.ckpt_dir = self.run_dir / "ckpt"
        
        if self.fabric.is_global_zero:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            # 保存配置文件
            self._save_config()
            
        self.seed = load_seed(self.config.seed)
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config, ddp=False)
        self.train_loader, self.test_loader = self.fabric.setup_dataloaders(self.train_loader, self.test_loader)

    def _save_config(self):
        """将配置对象保存为 yaml"""
        config_path = self.run_dir / "config.yaml"
        # 假设 config 是一个 Munch 对象或 Namespace，将其转为 dict
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else dict(self.config)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def train(self, resume=False):
        # 移除原来的 self.config.exp_name = ts 逻辑，统一使用 self.timestamp
        self.ckpt_prefix = "epoch" 

        # -------- Load models --------
        base_model = EquiformerV2(**self.config.model)
        self.model = CNF(base_model, self.config)

        if self.fabric.is_global_zero:
            print(f'Number of parameters: {count_parameters(self.model)}')
            print(f'Run Directory: {self.run_dir}')

        # -------- Optimizer & Scheduler --------
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr,
                                    weight_decay=self.config.train.weight_decay)
        
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.train.lr_decay)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        # -------- Resume Loading --------
        start_epoch = 0
        if resume and self.config.ckpt is not None:
            # 注意：这里的恢复逻辑可能需要根据你的新路径手动指定
            ckpt_path = Path(self.config.ckpt) 
            ckpt_dict = torch.load(ckpt_path, map_location=self.fabric.device)
            self.model.load_state_dict(ckpt_dict['state_dict'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.ema.load_state_dict(ckpt_dict['ema'])
            # 尝试从文件名解析 epoch
            try:
                start_epoch = int(ckpt_path.stem.split("_")[-1])
            except:
                start_epoch = 0
            if self.fabric.is_global_zero:
                print(f'Loaded checkpoint from {ckpt_path}')

        # -------- Logging Setup (Rank 0 only) --------
        if self.fabric.is_global_zero:
            # 日志存放在 run_dir/train.log
            log_file = self.run_dir / "train.log"
            self.logger = Logger(str(log_file), mode='a')
            start_log(self.logger, self.config)

            # Tensorboard 也可以统一放到这里
            writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

        self.loss_fn = CFMLoss(self.model, self.config)
        num_iter = 0

        # -------- Training Loop --------
        iter_wrapper = trange(start_epoch, self.config.train.num_epochs, desc='[Epoch]', position=1, leave=False) if self.fabric.is_global_zero else range(start_epoch, self.config.train.num_epochs)

        for epoch in iter_wrapper:
            self.train_chi = []
            self.model.train()
            start_time = time.time()

            for batch_idx, train_b in enumerate(self.train_loader):
                num_iter += 1
                self.optimizer.zero_grad()
                chi_loss = self.loss_fn(train_b)
                self.fabric.backward(chi_loss)

                if self.config.train.grad_norm > 0:
                    self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.config.train.grad_norm)

                self.optimizer.step()
                self.ema.update(self.model.parameters())
                self.train_chi.append(chi_loss.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            # -------- Validation --------
            self.model.eval()
            test_loss_chi = []
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):
                    chi_loss = self.loss_fn(test_b)
                    test_loss_chi.append(chi_loss.item())
            
            current_train_chi = np.mean(self.train_chi)
            current_test_chi = np.mean(test_loss_chi)

            if self.fabric.is_global_zero:
                writer.add_scalar("train_chi", current_train_chi, epoch)
                writer.add_scalar("test_chi", current_test_chi, epoch)
                
                self.logger.log(f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                               f'train chi: {current_train_chi:.3e} | '
                               f'test chi: {current_test_chi:.3e}', verbose=False)

                if (epoch + 1) % self.config.train.print_interval == 0:
                    tqdm.write(f'[EPOCH {epoch+1:04d}] | train: {current_train_chi:.3e} | test: {current_test_chi:.3e}')

                # -------- Save checkpoints --------
                if (epoch + 1) % self.config.train.save_interval == 0 or (epoch + 1) == self.config.train.num_epochs:
                    save_name = f'epoch_{epoch+1}.pth'
                    state = {
                        'epoch': epoch + 1,
                        'config': self.config,
                        'state_dict': self.model.state_dict(),
                        'ema': self.ema.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }
                    torch.save(state, self.ckpt_dir / save_name)

        return str(self.run_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    # Fabric
    parser.add_argument('--devices', default=4, type=int, help='Number of GPUs per node')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--strategy', default='auto', type=str, help='Strategy: auto, ddp, fsdp, etc.')
    parser.add_argument('--precision', default='32-true', type=str, help='Precision: 32-true, 16-mixed, etc.')

    args = parser.parse_args()

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision
    )
    fabric.launch()

    config = load_config(args.config, seed=args.seed)
    
    trainer = SidechainTrainer(config, fabric=fabric)
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()), resume=args.resume)