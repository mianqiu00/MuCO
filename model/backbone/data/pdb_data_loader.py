import functools as fn
import logging
import math
import os
import pickle
import random
import time
from functools import partial
from multiprocessing import get_context
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Optional

import lmdb
import numpy as np
import ot as pot
import pandas as pd
import torch
import torch.distributed as dist
import tree
from scipy.spatial.transform import Rotation
from torch.utils import data
from tqdm import tqdm

from model.FoldFlow2.data import utils as du
from model.FoldFlow2.flow.flow_utils.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat
from model.FoldFlow2.flow.flow_utils.so3_helpers import so3_relative_angle
from openfold.data import data_transforms
from openfold.utils import rigid_utils

_BYTES_PER_MEGABYTE = int(1e6)


def get_list_chunk_slices(lst, chunk_size):
    return [(i, i + chunk_size) for i in range(0, len(lst), chunk_size)]


def get_csv_rows_many(csv, shared_list, idx_slice):
    start_idx, end_idx = tuple(map(lambda x: min(x, len(csv)), idx_slice))
    for idx in tqdm(list(range(start_idx, end_idx))):
        shared_list[idx] = pickle.dumps(get_csv_row(csv, idx))

    print("Finished saving data to pickle")


def get_csv_row(csv, idx, is_linear=False):
    """Get on row of the csv file, and prepare the pdb feature dict.

    Args:
        idx (int): idx of the row
        csv (pd.DataFrame): csv pd.DataFrame

    Returns:
        tuple: dict of the features, ground truth backbone rigid, pdb_name
    """
    # Sample data example.
    example_idx = idx
    csv_row = csv.iloc[example_idx]
    if "pdb_name" in csv_row:
        pdb_name = csv_row["pdb_name"]
    elif "chain_name" in csv_row:
        pdb_name = csv_row["chain_name"]
    else:
        raise ValueError("Need chain identifier.")
    if is_linear:
        processed_file_path = csv_row["processed_path"].replace("cyclic", "linear").replace("data/yitian_wang", "home/yitian")
    else:
        processed_file_path = csv_row["processed_path"].replace("data/yitian_wang", "home/yitian")
    chain_feats = _process_csv_row(csv, processed_file_path, is_linear)

    if is_linear:
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["linear_rigidgroups_0"])[:, 0]
        return chain_feats, gt_bb_rigid, pdb_name, csv_row
    
    gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[:, 0]
    flowed_mask = np.ones_like(chain_feats["res_mask"])
    if np.sum(flowed_mask) < 1:
        raise ValueError("Must be flowed")
    fixed_mask = 1 - flowed_mask
    chain_feats["fixed_mask"] = fixed_mask
    chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
    chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

    return chain_feats, gt_bb_rigid, pdb_name, csv_row


def _process_csv_row(csv, processed_file_path, is_linear):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
    
    new_res_idx = torch.Tensor(new_res_idx).long()
    new_chain_idx = torch.Tensor(new_chain_idx).long()

    if is_linear:
        final_feats = {
            "linear_atom37_pos": chain_feats["all_atom_positions"], 
            "linear_rigidgroups_0": chain_feats["rigidgroups_gt_frames"]
        }
        return final_feats

    # To speed up processing, only take necessary features
    final_feats = {
        "aatype": chain_feats["aatype"],
        "seq_idx": new_res_idx,
        "chain_idx": new_chain_idx,
        "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
        "residue_index": processed_feats["residue_index"],
        "res_mask": processed_feats["bb_mask"],
        "atom37_pos": chain_feats["all_atom_positions"],
        "atom37_mask": chain_feats["all_atom_mask"],
        "atom14_pos": chain_feats["atom14_gt_positions"],
        "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
    }

    return final_feats


class PdbDataset(data.Dataset):
    """PDB dataset, with or without OT plan.

    Args:
        data_conf : configuration for the dataset
        gen_model : the model used to generate the data
        is_training : whether the dataset is used for training or validation
        is_OT : whether to use OT pairings
        ot_fn : method to use for OT (exact, sinkhorn). Default is `"exact"`.
        reg : regularization for Sinkhorn OT. Default is `0.05`.
        max_same_res : max number of same length proteins in a batch for OT. Default is `10`.
    """

    def __init__(
        self,
        *,
        data_conf,
        gen_model,
        is_training,
        is_OT=False,  # whether to use OT pairings
        ot_fn="exact",  # method to use for OT
        reg=0.05,  # regularization for OT
        max_same_res=10,  # max number of same length proteins in a batch for OT.
        is_sample=True,
        load_linear_data=False
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._data_split = data_conf.split
        self.is_sample = is_sample
        self.load_linear_data = load_linear_data
        self._init_metadata(data_conf.valid_num)

        self._cache_dataset = data_conf.cache_full_dataset
        self._cache_dataset_in_memory = data_conf.cache_dataset_in_memory
        self._cache_path = data_conf.cache_path
        self._store_result_tuples = None
        self._local_cache = None

        if self._cache_dataset:
            self._build_dataset_cache()

        # Could be Diffusion, CFM, OT-CFM or SF2M
        self._gen_model = gen_model
        self.is_OT = is_OT
        self.reg = reg
        self._max_same_res = max_same_res
        self._ot_fn = ot_fn.lower()

    @property
    def ot_fn(self):
        # import ot as pot
        if self._ot_fn == "exact":
            return pot.emd
        elif self._ot_fn == "sinkhorn":
            return partial(pot.sinkhorn, reg=self.reg)

    @property
    def max_same_res(self):
        if self.is_OT:
            return self._max_same_res
        else:
            return -1

    @property
    def is_training(self):
        return self._is_training

    @property
    def gen_model(self):
        return self._gen_model

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self, valid_num):
        """Initialize metadata."""
        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.max_len = int(pdb_csv["modeled_seq_len"].max())
        self.raw_csv = pdb_csv
    
        pdb_csv = pdb_csv[pdb_csv.type == "cyclic_aligned"]
        pdb_csv = pdb_csv[pdb_csv.cluster == self._data_split]

        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[: filter_conf.subset]

        pdb_csv = pdb_csv.sort_values("modeled_seq_len", ascending=False)
        self._create_split(pdb_csv, valid_num)

    def _build_dataset_cache(self):
        print(
            f"Starting to process dataset csv into memory "
            f"(cache_dataset_in_memory {self._cache_dataset_in_memory})"
        )
        print(f"ROWS {len(self.csv)}")
        # self.csv = self.csv.iloc[:500]
        print(f"Running only {len(self.csv)}")

        build_local_cache = True
        if os.path.isdir(self._cache_path):
            build_local_cache = False
            print(f"Found local cache @ {self._cache_path}, skipping build")

        # Initialize local cache with lmdb
        self._local_cache = lmdb.open(
            self._cache_path, map_size=(1024**3) * 60
        )  # 1GB * 60

        st_time = time.time()

        if build_local_cache:
            print(f"Building cache and saving @ {self._cache_path}")

            dataset_size = len(self.csv)
            num_chunks = math.ceil(
                float(dataset_size) / self.data_conf.num_csv_processors
            )

            idx_chunks = get_list_chunk_slices(list(range(dataset_size)), num_chunks)

            result_tuples = [None] * len(self.csv)

            pbar = tqdm(total=len(self.csv))
            with self._local_cache.begin(write=True) as txn:
                with SharedMemoryManager() as smm:
                    with get_context("spawn").Pool(
                        self.data_conf.num_csv_processors
                    ) as pool:
                        shared_list = smm.ShareableList(
                            [
                                bytes(3 * _BYTES_PER_MEGABYTE)
                                for _ in range(len(self.csv))
                            ]
                        )
                        partial_fxn = fn.partial(
                            get_csv_rows_many, self.csv, shared_list
                        )
                        iterator = enumerate(pool.imap(partial_fxn, idx_chunks))
                        for idx, _ in iterator:
                            start_idx, end_idx = tuple(
                                map(lambda x: min(x, len(self.csv)), idx_chunks[idx])
                            )
                            # print(f"RUNNING {start_idx} {end_idx} : chunks  {idx_chunks[idx]}")
                            for inner_idx in tqdm(range(start_idx, end_idx)):
                                txn.put(str(inner_idx).encode(), shared_list[inner_idx])

                                if self._cache_dataset_in_memory:
                                    result_tuples[inner_idx] = pickle.loads(
                                        shared_list[inner_idx]
                                    )

                                shared_list[inner_idx] = ""
                                pbar.update(1)
        elif self._cache_dataset_in_memory:
            print(f"Loading cache from local dataset @ {self._cache_path}")
            result_tuples = [None] * len(self.csv)
            with self._local_cache.begin() as txn:
                for ix in range(len(self.csv)):
                    result_tuples[ix] = pickle.loads(txn.get(str(ix).encode()))

        if self._cache_dataset_in_memory:

            def _get_list(idx):
                return list(map(lambda x: x[idx], result_tuples))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.chain_ftrs = _get_list(0)
            self.gt_bb_rigid_vals = _get_list(1)
            self.pdb_names = _get_list(2)
            self.csv_rows = _get_list(3)

        print(
            f"Finished processing dataset csv into memory in {time.time() - st_time} seconds"
        )
        print("Finished loading dataset into RAM")

    def _get_cached_csv_row(self, idx, csv=None):
        if csv is not None:
            # We are going to get the idx row out of the csv -> so we look for true index based on index cl
            idx = csv.iloc[idx]["index"]

        if self._cache_dataset_in_memory:
            return (
                self.chain_ftrs[idx],
                self.gt_bb_rigid_vals[idx],
                self.pdb_names[idx],
                self.csv_rows[idx],
            )
        else:
            return self._get_cached_csv_irow(idx)

    def _get_cached_csv_irow(self, idx, csv=None):

        with self._local_cache.begin() as txn:
            data = txn.get(str(idx).encode())
        return pickle.loads(data)

    def _create_split(self, pdb_csv, valid_num):
        rng = np.random.default_rng(42)
        shuffled_idx = rng.permutation(len(pdb_csv))
        
        split_point = len(pdb_csv) - valid_num
        
        train_idx = shuffled_idx[:split_point]
        val_idx = shuffled_idx[split_point:]

        if self.is_training:
            self.csv = pdb_csv.iloc[train_idx].reset_index(drop=True)
            self._log.info(f"Training: {len(self.csv)} examples ({(split_point / len(pdb_csv)) * 100:.1f}%)")
        else:
            if self.is_sample:
                self.csv = pdb_csv.iloc[val_idx].reset_index(drop=True)
            else:
                self.csv = pdb_csv
            self._log.info(f"Validation: {len(self.csv)} examples ({(len(self.csv) / len(pdb_csv)) * 100:.1f}%)")

    def __len__(self):
        return len(self.csv)

    def _get_csv_row(self, idx, csv=None, is_linear=False):
        """Get on row of the csv file, and prepare the pdb feature dict.

        Args:
            idx (int): idx of the row
            csv (pd.DataFrame): csv pd.DataFrame

        Returns:
            tuple: dict of the features, ground truth backbone rigid, pdb_name
        """
        if self._cache_dataset:
            return self._get_cached_csv_row(idx, csv)
        else:
            if csv is None:
                csv = self.csv

            return get_csv_row(csv, idx, is_linear)

    def __getitem__(self, idx) -> Any:
        # Custom sampler can return None for idx None.
        # Hacky way to simulate a fixed batch size.
        if idx is None:
            return None

        # print(f"[DEBUG] Train dataset getitem")
        chain_feats, gt_bb_rigid, pdb_name, _ = self._get_csv_row(idx)
        if self.load_linear_data:
            linear_chain_feats, linear_bb_rigid, _, _ = self._get_csv_row(idx, is_linear=True)
            chain_feats.update(linear_chain_feats)
        chain_feats = tree.map_structure(torch.Tensor, chain_feats)

        if self.is_training and not self.is_OT:
            # Sample t and flow.
            t = np.random.uniform(self._data_conf.min_t, 1.0)
            gen_feats_t = self._gen_model.forward_marginal(
                rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=linear_bb_rigid
            )
        elif self.is_training and self.is_OT:
            t = np.random.uniform(self._data_conf.min_t, 1.0)
            n_res = chain_feats["aatype"].shape[
                0
            ]  # feat['aatype'].shape = (batch, n_res)
            # get a maximum of self.max_same_res proteins with the same length
            subset = self.csv[self.csv["modeled_seq_len"] == n_res]
            n_samples = min(subset.shape[0], self.max_same_res)
            if n_samples == 1 or n_samples == 0:
                # only one sample, we can't do OT
                # self._log.info(f"Only one sample of length {n_res}, skipping OT")
                gen_feats_t = self._gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=None
                )
            else:
                sample_subset = subset.sample(
                    n_samples, replace=True, random_state=0
                ).reset_index(drop=True)

                # get the features, transform them to Rigid, and extract their translation and rotation.
                list_feat = [
                    self._get_csv_row(i, sample_subset)[0] for i in range(n_samples)
                ]
                list_trans_rot = [
                    extract_trans_rots_mat(
                        rigid_utils.Rigid.from_tensor_7(feat["rigids_0"])
                    )
                    for feat in list_feat
                ]
                list_trans, list_rot = zip(*list_trans_rot)

                # stack them and change them to torch.tensor
                sample_trans = torch.stack(
                    [torch.from_numpy(trans) for trans in list_trans]
                )
                sample_rot = torch.stack([torch.from_numpy(rot) for rot in list_rot])

                device = sample_rot.device  # TODO: set the device before that...

                # random matrices on S03.
                rand_rot = torch.tensor(
                    Rotation.random(n_samples * n_res).as_matrix()
                ).to(device=device, dtype=sample_rot.dtype)
                rand_rot = rand_rot.reshape(n_samples, n_res, 3, 3)
                # rand_rot_axis_angle = matrix_to_axis_angle(rand_rot)

                # random translation
                rand_trans = torch.randn(size=(n_samples, n_res, 3)).to(
                    device=device, dtype=sample_trans.dtype
                )

                # compute the ground cost for OT: sum of the cost for S0(3) and R3.
                ground_cost = torch.zeros(n_samples, n_samples).to(device)

                for i in range(n_samples):
                    for j in range(i, n_samples):
                        s03_dist = torch.sum(
                            so3_relative_angle(sample_rot[i], rand_rot[j])
                        )
                        r3_dist = torch.sum(
                            torch.linalg.norm(sample_trans[i] - rand_trans[j], dim=-1)
                        )
                        ground_cost[i, j] = s03_dist**2 + r3_dist**2
                        ground_cost[j, i] = ground_cost[i, j]

                # OT with uniform distributions over the set of pdbs
                a = pot.unif(n_samples, type_as=ground_cost)
                b = pot.unif(n_samples, type_as=ground_cost)
                T = self.ot_fn(
                    a, b, ground_cost
                )  # NOTE: `ground_cost` is the squared distance on SE(3)^N.

                # sample using the plan
                # pick one random indices for the pdb returned by __getitem__
                idx_target = torch.randint(n_samples, (1,))
                pi_target = T[idx_target].squeeze()
                pi_target /= torch.sum(pi_target)
                idx_source = torch.multinomial(pi_target, 1)
                paired_rot = rand_rot[idx_source].squeeze()
                paired_trans = rand_trans[idx_source].squeeze()

                rigids_1 = assemble_rigid_mat(paired_rot, paired_trans)

                gen_feats_t = self._gen_model.forward_marginal(
                    rigids_0=gt_bb_rigid, t=t, flow_mask=None, rigids_1=rigids_1
                )

        else:
            t = 1.0
            # gen_feats_t = self.gen_model.sample_ref(
            #     n_samples=gt_bb_rigid.shape[0],
            #     impute=gt_bb_rigid,
            #     flow_mask=None,
            #     as_tensor_7=False,
            # )
            # rigid_update = gen_feats_t["rigids_t"]
            # rigids_t =linear_bb_rigid.compose(rigid_update)
            rigids_t =linear_bb_rigid
            gen_feats_t = {}
            gen_feats_t["rigids_t"] = rigids_t.to_tensor_7()
        chain_feats.update(gen_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )
        final_feats = du.pad_feats(final_feats, self.max_len)
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class TrainSampler(data.Sampler):
    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        sample_mode,
        max_squared_res,
        num_gpus,
    ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self._max_squared_res = max_squared_res
        self.sampler_len = len(self._dataset_indices) * self._batch_size
        self._num_gpus = num_gpus

    def __iter__(self):
        # print(f"[DEBUG] Train sample")

        if self._sample_mode == "length_batch":
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby("modeled_seq_len").sample(
                self._batch_size, replace=True, random_state=self.epoch
            )
            return iter(sampled_order["index"].tolist())
        elif self._sample_mode == "time_batch":
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        else:
            random.shuffle(self._dataset_indices)
            return iter(self._dataset_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self._sample_mode == "default":
            return len(self._dataset_indices)
        else:
            return self.sampler_len


class DistributedTrainSampler(TrainSampler):
    """
    Takes in a rank arg for shuffling
    """

    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        sample_mode,
        rank,
        max_squared_res,
        num_gpus,
    ):
        self.rank = rank
        super().__init__(
            data_conf=data_conf,
            dataset=dataset,
            batch_size=batch_size,
            sample_mode=sample_mode,
            max_squared_res=max_squared_res,
            num_gpus=num_gpus,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
        # self.epoch = epoch + 123456 * self.rank
