import argparse
import time
import torch
from lightning.fabric import Fabric
from model.sidechain.utils.loader import load_config
from runner.sidechain_trainer import SidechainTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', action='store_true') # 改为 action_true 方便使用
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--devices', default=4, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--strategy', default='auto', type=str)
    parser.add_argument('--precision', default='32-true', type=str)

    args = parser.parse_args()

    # 初始化 Fabric
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision
    )
    fabric.launch()

    # --- 关键修改：多卡同步时间戳 ---
    if fabric.is_global_zero:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    else:
        timestamp = None

    # 将时间戳从 rank 0 广播到所有进程，确保文件夹名一致
    timestamp = fabric.broadcast(timestamp)

    config = load_config(args.config, seed=args.seed)
    
    trainer = SidechainTrainer(config, fabric=fabric, timestamp=timestamp)
    trainer.train(resume=args.resume)