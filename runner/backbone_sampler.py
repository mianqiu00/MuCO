"""Pytorch script for running Batch Inference with FoldFlow."""
import os
import copy
import logging
import time
from typing import Optional
from tqdm import tqdm

# Environment setup
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tree
from omegaconf import DictConfig

# Import your custom modules
from model.backbone.data import all_atom, pdb_data_loader
from model.backbone.data import utils as du
from model.backbone.flow import se3_fm
from model.backbone import flow_model, ff2_dependencies
from openfold.utils import rigid_utils as ru
from openfold.np.residue_constants import restype_atom37_mask
from openfold.np import residue_constants

from runner import experiments_utils as eu
try:
    from loss import metrics
except ModuleNotFoundError:
    metrics = None


def initialize_amino_acids(gt_prot, res_mask, psi_pred, rigid_traj, aatype):
    batch_size, num_res = gt_prot.shape[:2]
    device = gt_prot.device

    N_idx, CA_idx, C_idx, O_idx = 0, 1, 2, 4

    n_pos = gt_prot[..., N_idx, :]
    ca_pos = gt_prot[..., CA_idx, :]
    c_pos = gt_prot[..., C_idx, :]

    v1 = c_pos - ca_pos
    e1 = F.normalize(v1, dim=-1)

    v2 = n_pos - ca_pos
    
    u2 = v2 - torch.sum(v1 * v2, dim=-1, keepdim=True) * e1
    
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)

    rot_mat = torch.stack([e1, e2, e3], dim=-2) # [B, L, 3, 3]

    prot_centered = gt_prot - ca_pos.unsqueeze(-2) # [B, L, 37, 3]
    
    prot_local = torch.einsum('blij,blaj->blai', rot_mat, prot_centered)

    c_local = prot_local[..., C_idx, :] # [B, L, 3]
    o_local = prot_local[..., O_idx, :] # [B, L, 3]

    co_vec = o_local - c_local
    
    r_yz = torch.sqrt(co_vec[..., 1]**2 + co_vec[..., 2]**2 + 1e-8)
    
    sin_psi = psi_pred[..., 0] 
    cos_psi = psi_pred[..., 1] 
    norm = torch.sqrt(sin_psi**2 + cos_psi**2 + 1e-8)
    sin_psi = sin_psi / norm
    cos_psi = cos_psi / norm
    
    new_y = r_yz * cos_psi
    new_z =  - r_yz * sin_psi
    
    o_local_new = torch.stack([co_vec[..., 0], new_y, new_z], dim=-1) + c_local
    
    prot_local_mod = prot_local.clone()
    prot_local_mod[..., O_idx, :] = o_local_new

    rigid_traj = rigid_traj.unsqueeze(-2)
    rigids = ru.Rigid.from_tensor_7(rigid_traj) # [B, L]
    
    prot_final = rigids.apply(prot_local_mod) # [B, L, 37, 3]

    # res_mask: [B, L] -> [B, L, 37, 3]
    res_mask_expanded = res_mask[..., None, None].float()
    
    atom_mask = torch.tensor(restype_atom37_mask, device=device)[aatype]
    
    atom_mask_expanded = atom_mask[..., None].float()

    prot_final = prot_final * res_mask_expanded * atom_mask_expanded
    
    return prot_final


class BackboneSampler:
    def __init__(
        self,
        conf: DictConfig,
        ckpt_epoch: int,
        output_dir: str,
        split: Optional[str] = None, 
        device_id: int = 0,
        is_sample: bool = True,
        batch_size: Optional[int] = None,
    ):
        """Initialize Inference.

        Args:
            conf: Experiment configuration.
            ckpt_path: Path to the .pth checkpoint file.
            output_dir: Directory to save PDBs and metrics.
            device_id: GPU ID to use (default 0).
        """
        self._log = logging.getLogger(__name__)
        
        # 1. Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        self._fm_conf = conf.flow_matcher
        self._model_conf = conf.model
        self._data_conf = conf.data
        self.output_dir = output_dir
        self.is_sample = is_sample
        if batch_size is not None:
            self._exp_conf.batch_size = batch_size
        if split is not None:
            self._data_conf.split = split

        # 2. Setup Device (Single GPU)
        if torch.cuda.is_available() and device_id >= 0:
            self.device = torch.device(f"cuda:{device_id}")
            self._log.info(f"Using GPU: {self.device}")
            # Optimize settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
        else:
            self.device = torch.device("cpu")
            self._log.info("Using CPU")

        # 3. Load checkpoint before model construction so embedded ESM weights can
        # initialize the sequence encoder without touching external caches.
        full_ckpt_dir = self._exp_conf.full_ckpt_dir
        ckpt_path = full_ckpt_dir if os.path.isfile(full_ckpt_dir) else os.path.join(full_ckpt_dir, f"epoch_{ckpt_epoch}.pth")
        state_dict = self._read_checkpoint_state_dict(ckpt_path)

        # 4. Initialize Model and Flow Matcher
        # We need the flow matcher to initialize the dataset correctly
        self._flow_matcher = se3_fm.SE3FlowMatcher(self._fm_conf)
        esm_state_dict = self._extract_esm_state_dict(state_dict)
        dependencies = ff2_dependencies.FF2Dependencies(conf, esm_state_dict=esm_state_dict)
        self._model = flow_model.FF2Model.from_dependencies(dependencies)
        self._load_state_dict(state_dict)
        
        # Move to device and eval mode
        self._model = self._model.to(self.device)
        self._model.eval()
        self._model.float()

    def _read_checkpoint_state_dict(self, ckpt_path):
        """Reads weights, handling potential DDP prefix issues."""
        self._log.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if 'model' in checkpoint:
            return checkpoint['model']
        else:
            return checkpoint

    @staticmethod
    def _extract_esm_state_dict(state_dict):
        prefix = "seq_encoder.esm."
        esm_state_dict = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        return esm_state_dict or None

    def _load_state_dict(self, state_dict):
        # Load
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            self._log.warning(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            self._log.warning(f"Unexpected keys: {unexpected}")
        
        self._log.info("Checkpoint loaded successfully.")

    def _to_float32_tree(self, x):
        return tree.map_structure(
            lambda t: t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.float64 else t,
            x
        )

    def create_dataloader(self):
        """Creates the validation/inference dataloader."""
        # Note: Using is_training=False, but keeping OT logic if required by your loader
        test_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
            is_training=False, 
            is_sample=self.is_sample,
            load_linear_data=True
        )

        # Create loader (no distributed sampler needed)
        test_loader = du.create_data_loader(
            test_dataset,
            sampler=None, # Standard sequential sampling
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=0, # Avoid multiprocessing overhead for inference scripts often
            drop_last=False,
        )
        
        return test_loader

    def run_batch_inference(self):
        """Main loop: Iterates over data, runs inference, saves PDBs."""
        
        test_loader = self.create_dataloader()
        os.makedirs(self.output_dir, exist_ok=True)
        
        inference_metrics = []
        total_samples = 0
        
        self._log.info(f"Starting inference on {len(test_loader)} batches...")
        
        start_time = time.time()

        for batch_idx, (valid_feats, pdb_names) in enumerate(test_loader):
            
            # 1. Prepare Data
            # Convert boolean masks to numpy for post-processing logic
            res_mask = du.move_to_np(valid_feats["res_mask"].bool())
            fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
            aatype = du.move_to_np(valid_feats["aatype"])
            gt_prot = du.move_to_np(valid_feats["atom37_pos"])
            linear_prot = du.move_to_np(valid_feats["linear_atom37_pos"])
            batch_size = res_mask.shape[0]
            
            # Move tensors to GPU
            valid_feats = tree.map_structure(lambda x: x.to(self.device), valid_feats)
            valid_feats = self._to_float32_tree(valid_feats)

            # 2. Run Inference
            infer_out = self.inference_fn(
                valid_feats,
                noise_scale=self._exp_conf.noise_scale,
                batch_idx=batch_idx+1
            )

            # 3. Post-Process and Save
            # Assuming infer_out returns "prot_traj" where index 0 is final state
            # final_prot = infer_out["prot_traj"][0]
            rigid_traj = infer_out["rigid_traj"][0]
            psi_pred = infer_out["psi_pred"][0]

            final_prot = initialize_amino_acids(
                torch.Tensor(linear_prot).to(self.device), 
                torch.Tensor(res_mask).to(self.device), 
                torch.Tensor(psi_pred).to(self.device), 
                torch.Tensor(rigid_traj).to(self.device), 
                torch.Tensor(aatype).long().to(self.device)
            )
            final_prot = du.move_to_np(final_prot)

            for i in range(batch_size):
                pdb_name = pdb_names[i]
                num_res = int(np.sum(res_mask[i]).item())
                
                # Unpad based on masks
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_flow_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                
                percent_flowed = np.sum(unpad_flow_mask) / num_res
                
                # Clean filename
                safe_pdb_name = pdb_name.replace(".pdb", "").split("/")[-1]
                # save_name = f"{safe_pdb_name}_len{num_res}_flowed{percent_flowed:.2f}.pdb"
                save_name = f"{safe_pdb_name}.pdb"
                prot_path = os.path.join(self.output_dir, save_name)

                # Write PDB
                saved_path = eu.write_prot_to_pdb(
                    unpad_prot,
                    prot_path,
                    aatype=unpad_gt_aatype,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                    no_indexing=True
                )
                
                # 4. Compute Metrics (if possible)
                if metrics is None:
                    continue
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        flow_mask=unpad_flow_mask,
                    )
                    
                    sample_metrics["pdb_name"] = safe_pdb_name
                    sample_metrics["num_res"] = num_res
                    sample_metrics["flowed_percentage"] = percent_flowed
                    sample_metrics["sample_path"] = saved_path
                    inference_metrics.append(sample_metrics)
                    
                except ValueError as e:
                    self._log.warning(f"Failed metrics for {safe_pdb_name}: {e}")

            # 5. Save Summary CSV
            if inference_metrics:
                csv_path = os.path.join(self.output_dir, f"inference_metrics_{batch_idx}.csv")
                df = pd.DataFrame(inference_metrics)
                df.to_csv(csv_path, index=False)
                self._log.info(f"Saved metrics to {csv_path}")
        
        elapsed = time.time() - start_time
        self._log.info(f"Inference complete. Total time: {elapsed:.2f}s")

    def build_sequence_features(self, sequence: str, pad_len: int = None):
        """Build inference features directly from an amino-acid sequence."""
        sequence = sequence.strip().upper()
        if not sequence:
            raise ValueError("Input sequence is empty.")

        allowed = set(residue_constants.restypes_with_x)
        bad = sorted(set(sequence) - allowed)
        if bad:
            raise ValueError(f"Unsupported residue letters in sequence: {''.join(bad)}")

        aatype = np.asarray(
            [residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in sequence],
            dtype=np.int64,
        )
        n_res = len(sequence)
        res_mask = np.ones(n_res, dtype=np.float32)
        fixed_mask = np.zeros(n_res, dtype=np.float32)
        residue_index = np.arange(1, n_res + 1, dtype=np.int64)

        # Template backbone used only for OpenFold-style feature completeness and
        # O-atom reconstruction; the SE(3) reference state is still sampled.
        rigids_0 = ru.Rigid.identity(
            (n_res,), dtype=torch.float32, device=torch.device("cpu"), requires_grad=False
        )
        psi = torch.zeros((n_res, 2), dtype=torch.float32)
        psi[:, 1] = 1.0
        linear_atom37_pos = all_atom.compute_backbone(rigids_0, psi)[0].numpy()
        atom37_mask = restype_atom37_mask[aatype].astype(np.float32)
        atom37_pos = linear_atom37_pos * atom37_mask[..., None]

        chain_feats = {
            "aatype": aatype,
            "seq_idx": residue_index,
            "chain_idx": np.ones(n_res, dtype=np.int64),
            "residue_index": residue_index,
            "res_mask": res_mask,
            "fixed_mask": fixed_mask,
            "atom37_pos": atom37_pos.astype(np.float32),
            "atom37_mask": atom37_mask,
            "linear_atom37_pos": linear_atom37_pos.astype(np.float32),
            "sc_ca_t": np.zeros((n_res, 3), dtype=np.float32),
        }

        gen_feats_t = self._flow_matcher.sample_ref(
            n_samples=n_res,
            impute=rigids_0,
            flow_mask=None,
            as_tensor_7=True,
        )
        chain_feats.update(gen_feats_t)
        chain_feats["t"] = 1.0

        tensor_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        padded_feats = du.pad_feats(tensor_feats, pad_len or n_res)
        padded_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), padded_feats
        )
        return padded_feats, n_res

    def sample_sequence_to_pdb(self, sequence: str, pdb_name: str, output_dir: str = None):
        """Generate one backbone PDB directly from a sequence."""
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        feats, n_res = self.build_sequence_features(sequence)
        res_mask = feats["res_mask"].bool().numpy()[None]
        fixed_mask = feats["fixed_mask"].bool().numpy()[None]
        aatype = feats["aatype"].numpy()[None]
        linear_prot = feats["linear_atom37_pos"].numpy()[None]

        batched_feats = tree.map_structure(
            lambda x: x.unsqueeze(0).to(self.device) if torch.is_tensor(x) else x,
            feats,
        )
        batched_feats = self._to_float32_tree(batched_feats)

        infer_out = self.inference_fn(
            batched_feats,
            noise_scale=self._exp_conf.noise_scale,
            batch_idx=1,
            aux_traj=False,
            precompute_seq=True,
        )
        rigid_traj = du.move_to_np(infer_out["rigids_final"])
        psi_pred = infer_out["psi_pred"][0]

        final_prot = initialize_amino_acids(
            torch.tensor(linear_prot).to(self.device),
            torch.tensor(res_mask).to(self.device),
            torch.tensor(psi_pred).to(self.device),
            torch.tensor(rigid_traj).to(self.device),
            torch.tensor(aatype).long().to(self.device),
        )
        final_prot = du.move_to_np(final_prot)[0][res_mask[0]]
        unpad_fixed_mask = fixed_mask[0][res_mask[0]]
        unpad_aatype = aatype[0][res_mask[0]]

        safe_name = os.path.basename(pdb_name).replace(".pdb", "")
        prot_path = os.path.join(output_dir, f"{safe_name}.pdb")
        return eu.write_prot_to_pdb(
            final_prot,
            prot_path,
            aatype=unpad_aatype,
            b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
            no_indexing=True,
        )

    def sample_sequences_to_pdb(self, sequences, pdb_names, output_dir: str = None, progress_callback=None):
        """Generate backbone PDBs for same-length sequences in one batch."""
        if len(sequences) != len(pdb_names):
            raise ValueError("sequences and pdb_names must have the same length")
        if not sequences:
            return []
        pad_len = max(len(seq.strip()) for seq in sequences)

        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        feats_list = [self.build_sequence_features(seq, pad_len=pad_len)[0] for seq in sequences]
        batched_feats = tree.map_structure(lambda *xs: torch.stack(xs, dim=0), *feats_list)
        res_mask = batched_feats["res_mask"].bool().numpy()
        fixed_mask = batched_feats["fixed_mask"].bool().numpy()
        aatype = batched_feats["aatype"].numpy()
        linear_prot = batched_feats["linear_atom37_pos"].numpy()

        device_feats = tree.map_structure(lambda x: x.to(self.device), batched_feats)
        device_feats = self._to_float32_tree(device_feats)

        infer_out = self.inference_fn(
            device_feats,
            noise_scale=self._exp_conf.noise_scale,
            batch_idx=1,
            aux_traj=False,
            precompute_seq=True,
            progress_callback=progress_callback,
        )
        rigid_traj = du.move_to_np(infer_out["rigids_final"])
        psi_pred = infer_out["psi_pred"][0]

        final_prot = initialize_amino_acids(
            torch.tensor(linear_prot).to(self.device),
            torch.tensor(res_mask).to(self.device),
            torch.tensor(psi_pred).to(self.device),
            torch.tensor(rigid_traj).to(self.device),
            torch.tensor(aatype).long().to(self.device),
        )
        final_prot = du.move_to_np(final_prot)

        paths = []
        for i, pdb_name in enumerate(pdb_names):
            unpad_prot = final_prot[i][res_mask[i]]
            unpad_fixed_mask = fixed_mask[i][res_mask[i]]
            unpad_aatype = aatype[i][res_mask[i]]
            safe_name = os.path.basename(pdb_name).replace(".pdb", "")
            prot_path = os.path.join(output_dir, f"{safe_name}.pdb")
            paths.append(eu.write_prot_to_pdb(
                unpad_prot,
                prot_path,
                aatype=unpad_aatype,
                b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                no_indexing=True,
            ))
        return paths
    
    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self._flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats
    
    def _self_conditioning(self, batch):
        model_sc = self._model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=True,
        self_condition=True,
        noise_scale=1.0,
        context=None,
        batch_idx=0,
        precompute_seq=False,
        progress_callback=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if precompute_seq:
            seq_emb_s, seq_emb_z = self._model.precompute_sequence_repr(sample_feats)
            sample_feats["seq_emb_s"] = seq_emb_s
            sample_feats["seq_emb_z"] = seq_emb_z
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
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))] if aux_traj else []
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        final_psi = None
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self._self_conditioning(sample_feats)
            for step_idx, t in enumerate(tqdm(reverse_steps, desc=f"Processing inference steps on batch {batch_idx}"), start=1):

                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self._model(sample_feats)
                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                rots_t, trans_t, rigids_t = self._flow_matcher.reverse(
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
                final_psi = psi_pred
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                if aux_traj:
                    atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                    all_bb_prots.append(du.move_to_np(atom37_t))
                if progress_callback is not None:
                    progress_callback(step_idx, len(reverse_steps))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots) if aux_traj else None
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
        else:
            ret["rigids_final"] = sample_feats["rigids_t"]
            ret["psi_pred"] = final_psi[None]
        return ret
