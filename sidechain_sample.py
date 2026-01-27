import time
import argparse
from model.sidechain.utils.loader import load_config
from runner.sidechain_sampler import SidechainSampler


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='sidechain_sample.yaml')
    parser.add_argument('name', type=str, default='test')
    parser.add_argument('--save_traj', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_gt_masks', type=bool, default=False)
    parser.add_argument('--inpaint', type=str, default='')

    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    sampler = SidechainSampler(config, args.use_gt_masks)
    sampler.sample(time.strftime('%b%d-%H:%M:%S', time.gmtime()), name=args.name, save_traj=args.save_traj, inpaint=args.inpaint)
    print(f'Inference took a total of {time.time() - start_time} seconds')

    # python sampler_pdb.py base CPSea_PDB