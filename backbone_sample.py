import argparse
import logging
from omegaconf import OmegaConf
from runner.backbone_sampler import BackboneSampler
import os
from utils import seed_everything, pdb_to_pickle

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")


def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_timestamp", type=str, default="2025-12-03_12-52-37", help="Path to config.yaml")
    parser.add_argument("--ckpt_epoch", type=int, default=100, help="Path to model checkpoint.pth")
    parser.add_argument("--output", type=str, default="./data/inference/pdb/coarse", help="Output directory")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=1024, help="Override batch size")
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--split", type=str, default="CPBind")
    args = parser.parse_args()

    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load Config
    conf_path = os.path.join("./saved_model", args.config_timestamp, "config.yaml")
    conf = OmegaConf.load(conf_path)

    # Optional: Override config values via CLI
    if args.batch_size:
        conf.experiment.eval_batch_size = args.batch_size
    if args.split:
        conf.data.split = args.split

    split = conf.data.split
    output_path_pdb = args.output + "_" + split 
    if not os.path.exists(output_path_pdb):
        os.makedirs(output_path_pdb, exist_ok=True)

        # Initialize and Run
        runner = BackboneSampler(
            conf=conf,
            ckpt_epoch=args.ckpt_epoch,
            output_dir=output_path_pdb,
            device_id=args.device, 
            is_sample=args.sample,
            batch_size=args.batch_size
        )
        runner.run_batch_inference()

    else:
        print("Output path already exists. Skipping inference.")

    output_path_pkl = output_path_pdb.replace("pdb", "pkl")
    if not os.path.exists(output_path_pkl):
        os.makedirs(output_path_pkl, exist_ok=True)
    pdb_to_pickle(output_path_pdb, output_path_pkl)

if __name__ == "__main__":
    main()