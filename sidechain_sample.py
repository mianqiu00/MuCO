import time
import argparse
import torch
import os
from ml_collections import ConfigDict
from runner.sidechain_sampler import SidechainSampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts', type=str, required=True, help='Checkpoint timestamp')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--base_dir', type=str, default='./sample', help='Base dir for ckpts')
    
    parser.add_argument('--name', type=str, default='test_run')
    parser.add_argument('--test_path', type=str, default=None, help='Override test data path')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--coeff', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--save_traj', action='store_true')
    parser.add_argument('--use_gt_masks', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = get_args()

    ckpt_path = os.path.join(args.base_dir, f"{args.ts}_{args.epoch}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path}")

    print(f"Loading config from {ckpt_path}...")
    ckpt_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt_dict['config']
    
    config.ckpt = ckpt_path
    config.seed = args.seed
    
    if not hasattr(config, 'sample'):
        config.sample = ConfigDict()
    
    config.sample.n_samples = args.n_samples
    config.sample.num_steps = args.num_steps
    config.sample.coeff = args.coeff
    
    if args.test_path:
        config.data.test_path = args.test_path

    sampler = SidechainSampler(config, args.use_gt_masks)
    
    sampler.sample(
        sample_name=args.name, 
        save_traj=args.save_traj
    )
    
    print(f'Inference took a total of {time.time() - start_time:.2f} seconds')