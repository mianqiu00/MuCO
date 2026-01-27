"""Pytorch script for training backbone."""
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import copy
import logging
import time
from datetime import datetime
from collections import defaultdict

import GPUtil
import numpy as np
import pandas as pd
import torch
import tree
from einops import rearrange
import runner.experiments_utils as eu
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from model.backbone.flow.flow_utils.so3_helpers import hat_inv, pt_to_identity
from model.backbone.data import all_atom, pdb_data_loader
from model.backbone.data import utils as du
from model.backbone.flow import se3_fm
from model.backbone import flow_model, ff2_dependencies
from openfold.utils import rigid_utils as ru

from loss import metrics
from utils import time_format


class BackboneTrainer:
    def __init__(
        self,
        *,
        conf: DictConfig,
    ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self.first_batch = None
        self._log = logging.getLogger(__name__)
        self._available_gpus = "".join(
            [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
        )

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        self._fm_conf = conf.flow_matcher
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_ddp = self._exp_conf.use_ddp
        self.trained_epochs = 0
        self.trained_steps = 0
        
        # 1. initialize ddp info if in ddp mode
        print(f"Number of threads {self._exp_conf.torch_num_threads}")
        torch.set_num_threads(self._exp_conf.torch_num_threads)
        # reduce matmul precision for better performance on GPU
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._master_proc = True

        from lightning.fabric.strategies import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=True)
        self.fabric = Fabric(
            accelerator="cuda", devices=self._exp_conf.num_gpus, strategy=strategy
        )
        self.fabric.launch()

        torch.backends.cuda.matmul.allow_tf32 = True
        self._log.info(f"Using DDP with {self.fabric.global_rank} rank")

        # 2. silent rest of logger when use ddp mode
        self._master_proc = self.fabric.global_rank == 0
        self._global_rank = self.fabric.global_rank
        print(
            f"RANK: {self.fabric.global_rank} | master process: {self._master_proc}"
        )

        if self.fabric.global_rank != 0:
            self._log.addHandler(logging.NullHandler())
            self._log.setLevel("ERROR")

        if self._use_ddp and self.fabric.global_rank != 0:
            self._exp_conf.full_ckpt_dir = None

        # 3. Initialize experiment objects
        self._flow_matcher = se3_fm.SE3FlowMatcher(self._fm_conf)
        dependencies = ff2_dependencies.FF2Dependencies(conf)
        self._model = flow_model.FF2Model.from_dependencies(dependencies)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f"Number of model parameters {num_parameters}")
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate
        )
        
        # 4. Initialize ckpt and eval direction
        if self._master_proc:
            time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Always start from base_dir
            base_dir = self._exp_conf.base_dir
            os.makedirs(base_dir, exist_ok=True)

            # Create run directory: saved_model/<timestamp>/
            run_dir = os.path.join(base_dir, time_now)
            os.makedirs(run_dir, exist_ok=True)

            # ---- CKPT ----
            ckpt_dir = os.path.join(run_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.full_ckpt_dir = ckpt_dir
            self._log.info(f"Checkpoints saved to: {ckpt_dir}")

            # ---- EVAL ----
            eval_dir = os.path.join(run_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f"Evaluation saved to: {eval_dir}")
        else:
            self._exp_conf.full_ckpt_dir = None
            self._exp_conf.eval_dir = None

        if self._master_proc:
            # ---- SAVE CONFIG ----
            import yaml
            config_path = os.path.join(run_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(OmegaConf.to_container(self._conf, resolve=True), f)
            self._log.info(f"Saved config.yaml to {config_path}")

            # ---- FILE LOGGING ----
            log_file = os.path.join(run_dir, "train.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self._log.addHandler(file_handler)
            self._log.info(f"Logging to file: {log_file}")

    @property
    def flow_matcher(self):
        return self._flow_matcher

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        # Loading dataset
        train_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_training=True,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
            load_linear_data=True
        )

        valid_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
            is_training=False,
            load_linear_data=True
        )

        # Loading sampler
        train_sampler = pdb_data_loader.DistributedTrainSampler(
            data_conf=self._data_conf,
            dataset=train_dataset,
            batch_size=self._exp_conf.batch_size,
            sample_mode=self._exp_conf.sample_mode,
            rank=self.fabric.global_rank,
            max_squared_res=self._exp_conf.max_squared_res,
            num_gpus=self._exp_conf.num_gpus,
        )
        valid_sampler = None

        # Loading data loader
        num_workers = self._exp_conf.num_loader_workers

        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )

        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        train_loader = self.fabric.setup_dataloaders(
            train_loader, use_distributed_sampler=False
        )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def start_training(self, return_logs=False):
        # print(f"Start-training-"*10)
        # Set environment variables for which GPUs to use.
        replica_id = 0
        assert not self._exp_conf.use_ddp or self._exp_conf.use_gpu

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus == 1:
                try:
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                except IndexError:
                    device = "cuda:0"
                    self._log.warning("Error on available gpus, trying with device 0")
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            # muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                # DDP mode
                self._model, self._optimizer = self.fabric.setup(
                    self._model, self._optimizer
                )
                device = self.fabric.device
                self._log.info(f"Using device: {device}")
        else:
            device = "cpu"
            self._log.info(f"Using device: {device}")
            self._model = self.model.to(device)

        # Loading data
        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
        ) = self.create_dataset()

        # Training loop
        self._model.train()

        logs = []
        self.best_loss = 1e10
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader, valid_loader, device, return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info("Done")
        return logs

    def update_fn(self, data, debug=False):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()

        loss, aux_data = self.loss_fn(data)
        self.fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        """Train for one epoch — only print epoch-level train loss and run one validation."""

        log_losses = defaultdict(list)
        epoch_log_losses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        epoch_start_time = log_time

        epoch_step = 0
        # ---- Training Loop (per sample) ----
        for train_feats in train_loader:

            # Move to device
            train_feats = tree.map_structure(lambda x: x.to(device), train_feats)

            # Update
            loss, aux_data = self.update_fn(train_feats)
            if torch.isnan(loss):
                raise Exception("NaN in loss encountered")

            # record logs
            if return_logs:
                global_logs.append(loss)

            for k, v in aux_data.items():
                if k in ['total_loss', 'rot_loss', 'trans_loss', 'bb_atom_loss', 'dist_mat_loss']:
                    log_losses[k].append(du.move_to_np(v))
                    epoch_log_losses[k].append(du.move_to_np(v))

            self.trained_steps += 1
            epoch_step += 1

            # step log
            if (
                self.trained_steps == 1
                or self.trained_steps % self._exp_conf.log_freq == 0
            ):
                elapsed_time = time.time() - log_time
                log_time = time.time()
                sec_per_step = elapsed_time / epoch_step
                epoch_step = 0
                rolling_losses = tree.map_structure(lambda xs: float(np.mean(np.asarray(xs, dtype=np.float64))), log_losses)
                loss_log = " ".join(
                    [
                        f"{k}={float(np.mean(v)):.4f}"
                        for k, v in rolling_losses.items()
                        if "batch" not in k
                    ]
                )
                self._log.info(
                    f"[Step {self.trained_steps}]: {loss_log}, sec/step={sec_per_step:.2f}"
                )
                log_losses = defaultdict(list)

        # ---- END of epoch — Print epoch-level average loss ----
        elapsed_time = time_format.format_seconds(time.time() - epoch_start_time)
        rolling_losses = tree.map_structure(lambda xs: float(np.mean(np.asarray(xs, dtype=np.float64))), epoch_log_losses)
        loss_log = " ".join(
            [
                f"{k}={float(np.mean(v)):.4f}" 
                for k, v in rolling_losses.items() 
                if "batch" not in k
            ]
        )
        self._log.info(f"[Epoch {self.trained_epochs + 1}] Train {loss_log}, sec/epoch={elapsed_time}")

        # ---- Run full validation ONCE per epoch ----
        if self._master_proc:
            eval_time = time.time()
            eval_dir = os.path.join(
                self._exp_conf.eval_dir, f"epoch_{self.trained_epochs + 1}"
            )
            os.makedirs(eval_dir, exist_ok=True)

            self._log.info(
                f"Running evaluation for full epoch {self.trained_epochs + 1} in {eval_dir}"
            )

            self.eval_fn(
                eval_dir,
                valid_loader,
                device,
                noise_scale=self._exp_conf.noise_scale,
            )

            # ---- END of evaluation — Print evaluation time ----
            elapsed_time = time_format.format_seconds(time.time() - eval_time)
            self._log.info(f"Finished evaluation in {elapsed_time}")

        # ---- Save checkpoint ONCE per epoch ----
        epoch_total_loss = float(np.mean(rolling_losses['total_loss']))
        if self._master_proc and self._exp_conf.full_ckpt_dir is not None and (epoch_total_loss < self.best_loss):
            ckpt_path = os.path.join(
                self._exp_conf.full_ckpt_dir, f"epoch_{self.trained_epochs + 1}.pth"
            )
            du.write_checkpoint(
                ckpt_path,
                self.model.state_dict(),
                self._conf,
                self._optimizer.state_dict(),
                self.trained_epochs + 1,
                self.trained_steps,
                logger=self._log,
                use_torch=True,
            )
        
        self.best_loss = min(epoch_total_loss, self.best_loss)

        if return_logs:
            return global_logs

    def eval_fn(
        self,
        eval_dir,
        valid_loader,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        context=None,
    ):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            # Move testing feats to device
            res_mask = du.move_to_np(valid_feats["res_mask"].bool())
            fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
            aatype = du.move_to_np(valid_feats["aatype"])
            gt_prot = du.move_to_np(valid_feats["atom37_pos"])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                context=context,
            )
            final_prot = infer_out["prot_traj"][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_flow_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_flowed = np.sum(unpad_flow_mask) / num_res
                prot_path = os.path.join(
                    eval_dir,
                    f"len_{num_res}_sample_{i}_{self.fabric.device}_flowed_{percent_flowed:.2f}.pdb",
                )

                # Extract argmax predicted aatype
                saved_path = eu.write_prot_to_pdb(
                    unpad_prot,
                    prot_path,
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        flow_mask=unpad_flow_mask,
                    )
                except ValueError as e:
                    self._log.warning(
                        f"Failed evaluation of length {num_res} sample {i}: {e}"
                    )
                    continue
                sample_metrics["step"] = self.trained_steps
                sample_metrics["num_res"] = num_res
                sample_metrics["fixed_residues"] = np.sum(unpad_fixed_mask)
                sample_metrics["flowed_percentage"] = percent_flowed
                sample_metrics["sample_path"] = saved_path
                sample_metrics["gt_pdb"] = pdb_names[i]
                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, "metrics.csv")
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        # Self Conditioning
        if self._model_conf.embed.embed_self_conditioning:
            batch["sc_ca_t"] = batch["rigids_t"][..., 4:].clone()
        if (
            self._model_conf.embed.embed_self_conditioning
            and self.trained_steps % 2 == 1
        ):
            with torch.no_grad():
                batch = self._self_conditioning(batch)

        _, gt_rot_u_t = self._flow_matcher._so3_fm.vectorfield(
            batch["rot_vectorfield"], batch["rot_t"], batch["t"]
        )

        # Model out process
        model_out = self.model(batch)
        bb_mask = batch["res_mask"]
        flow_mask = 1 - batch["fixed_mask"]
        loss_mask = bb_mask * flow_mask
        batch_size, num_res = bb_mask.shape

        gt_trans_u_t = batch["trans_vectorfield"]
        rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
        trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_v_t = model_out["rot_vectorfield"] * flow_mask[..., None, None]
        pred_trans_v_t = model_out["trans_vectorfield"] * flow_mask[..., None]

        # Translation vectorfield loss
        trans_vectorfield_mse = (gt_trans_u_t - pred_trans_v_t) ** 2 * loss_mask[
            ..., None
        ]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse / trans_vectorfield_scaling[:, None, None] ** 2,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out["rigids"][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_vectorfield_loss * (
            batch["t"] > self._exp_conf.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self._exp_conf.trans_x0_threshold)
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._fm_conf.flow_trans)

        # Rotation loss
        # gt_rot_u_t and pred_rot_v_t are matrices convert
        t_shape = batch["rot_t"].shape[0]
        rot_t = rearrange(batch["rot_t"], "t n c d -> (t n) c d", c=3, d=3).double()
        gt_rot_u_t = rearrange(gt_rot_u_t, "t n c d -> (t n) c d", c=3, d=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "t n c d -> (t n) c d", c=3, d=3)
        try:
            rot_t = rot_t.double()
            gt_at_id = pt_to_identity(rot_t, gt_rot_u_t)
            gt_rot_u_t = hat_inv(gt_at_id)
            pred_at_id = pt_to_identity(rot_t, pred_rot_v_t)
            pred_rot_v_t = hat_inv(pred_at_id)
        except ValueError as e:
            self._log.info(
                f"Skew symmetric error gt {((gt_at_id + gt_at_id.transpose(-1, -2))**2).mean()} "
                f"pred {((pred_at_id + pred_at_id.transpose(-1, -2))**2).mean()} Skipping rot loss"
            )
            gt_rot_u_t = torch.zeros_like(rot_t[..., 0])
            pred_rot_v_t = torch.zeros_like(rot_t[..., 0])

        gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_u_t, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_u_t / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_v_t, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_v_t / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_vectorfield_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_u_t - pred_rot_v_t) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
                dim=(-1, -2),
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._fm_conf.flow_rot)

        # Backbone atom loss
        pred_atom37 = model_out["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self._exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self.flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
        context=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]
        # dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning:
                sample_feats["sc_ca_t"] = sample_feats["rigids_t"][..., 4:].clone()
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:

                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                rots_t, trans_t, rigids_t = self.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret
