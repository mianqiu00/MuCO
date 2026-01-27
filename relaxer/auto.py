#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import os
import math
import random
import numpy as np
from openmm import app as openmm_app
from openmm import unit

from head_tail import ForceFieldMinimizerHeadTail
from k_to_de import ForceFieldMinimizerKtoDE
from cys_to_cys import ForceFieldMinimizerCys


class ForceFieldMinimizerAuto:
    """
    Automatically determine cyclization mode (head-tail, Cys-Cys, Lys-Asp/Glu)
    and perform energy minimization with distance-based validation.
    """

    # ==============================
    # User-defined cyclization cutoff (standard distance + 0.1A)
    # ==============================
    HEAD_TAIL_THRESHOLD = 1.43
    K_TO_DE_THRESHOLD = 1.43
    CYS_TO_CYS_THRESHOLD = 2.15


    def __init__(self, platform='CUDA', stiffness=10.0, log=False):
        """
        Parameters
        ----------
        platform : str
            OpenMM platform name ('CUDA', 'CPU', etc.)
        stiffness : float
            Harmonic restraint stiffness
        seed : int
            Random seed for reproducibility
        """
        self.platform = platform
        self.stiffness = stiffness
        self.log = log

        # Initialize all minimizers ONCE (heavy objects)
        self._runner_head_tail = ForceFieldMinimizerHeadTail(
            stiffness=self.stiffness,
            platform=self.platform
        )
        self._runner_k_to_de = ForceFieldMinimizerKtoDE(
            stiffness=self.stiffness,
            platform=self.platform
        )
        self._runner_cys = ForceFieldMinimizerCys(
            stiffness=self.stiffness,
            platform=self.platform
        )

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def _calc_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two OpenMM positions (Å)."""
        p1 = pos1.value_in_unit(unit.angstroms)
        p2 = pos2.value_in_unit(unit.angstroms)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _get_atom_pos(self, residue, atom_name, positions):
        """Return the position of a specific atom in a residue."""
        for atom in residue.atoms():
            if atom.name == atom_name:
                return positions[atom.index]
        return None

    def _calc_atom_distance_from_pdb(self, pdb_file, chain_id,
                                     res_idx1, atom1,
                                     res_idx2, atom2):
        """
        Calculate distance between two atoms from a PDB file (Å).
        Residue indices are chain-local (0-based).
        """
        pdb = openmm_app.PDBFile(pdb_file)
        positions = pdb.positions
        topology = pdb.topology

        chain = next(c for c in topology.chains() if c.id == chain_id)
        residues = list(chain.residues())

        pos1 = self._get_atom_pos(residues[res_idx1], atom1, positions)
        pos2 = self._get_atom_pos(residues[res_idx2], atom2, positions)

        if pos1 is None or pos2 is None:
            return None

        return self._calc_distance(pos1, pos2)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def __call__(self, input_pdb, output_pdb, log=False):
        if log:
            self.log = log
        # Parse PDB for fast geometric inspection
        pdb = openmm_app.PDBFile(input_pdb)
        positions = pdb.positions
        topology = pdb.topology

        chain = next(topology.chains())
        chain_id = chain.id
        residues = list(chain.residues())

        if len(residues) < 2:
            raise ValueError("Chain too short for cyclization.")

        head_res = residues[0]
        tail_res = residues[-1]

        res_idx_head = 0
        res_idx_tail = len(residues) - 1
        cyclic_opts = [((chain_id, res_idx_head), (chain_id, res_idx_tail))]
        cyclic_chains = [chain_id]

        h_name = head_res.name
        t_name = tail_res.name

        # Default values
        mode = "head_tail"
        dist_pre = 999.9
        bond_atom_head = None
        bond_atom_tail = None
        dist_desc = ""

        # Backbone distance (N-C)
        pos_n = self._get_atom_pos(head_res, 'N', positions)
        pos_c = self._get_atom_pos(tail_res, 'C', positions)

        dist_backbone = 999.9
        if pos_n and pos_c:
            dist_backbone = self._calc_distance(pos_n, pos_c)

        # --------------------------------------------------------------
        # Cys–Cys disulfide
        # --------------------------------------------------------------
        if h_name == 'CYS' and t_name == 'CYS':
            pos_sg1 = self._get_atom_pos(head_res, 'SG', positions)
            pos_sg2 = self._get_atom_pos(tail_res, 'SG', positions)

            if pos_sg1 and pos_sg2:
                dist_ss = self._calc_distance(pos_sg1, pos_sg2)
                if dist_ss < 2.5:
                    mode = "cys_to_cys"
                    dist_pre = dist_ss
                    bond_atom_head = 'SG'
                    bond_atom_tail = 'SG'
                    dist_desc = "SG - SG"
                else:
                    dist_pre = dist_backbone
                    bond_atom_head = 'N'
                    bond_atom_tail = 'C'
                    dist_desc = "N - C (Backbone)"
            else:
                dist_pre = dist_backbone
                bond_atom_head = 'N'
                bond_atom_tail = 'C'
                dist_desc = "N - C (Missing SG)"

        # --------------------------------------------------------------
        # Lys–Asp/Glu isopeptide
        # --------------------------------------------------------------
        elif (h_name == 'LYS' and t_name in ['ASP', 'GLU']) or \
             (h_name in ['ASP', 'GLU'] and t_name == 'LYS'):

            atom_h = 'NZ' if h_name == 'LYS' else ('CG' if h_name == 'ASP' else 'CD')
            atom_t = 'NZ' if t_name == 'LYS' else ('CG' if t_name == 'ASP' else 'CD')

            pos1 = self._get_atom_pos(head_res, atom_h, positions)
            pos2 = self._get_atom_pos(tail_res, atom_t, positions)

            dist_iso = 999.9
            if pos1 and pos2:
                dist_iso = self._calc_distance(pos1, pos2)

            if dist_iso < dist_backbone:
                mode = "k_to_de"
                dist_pre = dist_iso
                bond_atom_head = atom_h
                bond_atom_tail = atom_t
                dist_desc = f"{atom_h} - {atom_t} (Isopeptide)"
            else:
                dist_pre = dist_backbone
                bond_atom_head = 'N'
                bond_atom_tail = 'C'
                dist_desc = "N - C (Backbone shorter)"

        # --------------------------------------------------------------
        # Default head–tail
        # --------------------------------------------------------------
        else:
            dist_pre = dist_backbone
            bond_atom_head = 'N'
            bond_atom_tail = 'C'
            dist_desc = "N - C (Default)"

        # Select runner
        if mode == "cys_to_cys":
            runner = self._runner_cys
        elif mode == "k_to_de":
            runner = self._runner_k_to_de
        else:
            runner = self._runner_head_tail

        if self.log:
            print(f"\n[AutoCycler] Mode: {mode}")
            print(f"[AutoCycler] Pair: {h_name} - {t_name}")
            print(f"[AutoCycler] Bond atoms: {bond_atom_head} - {bond_atom_tail}")
            print(f"[AutoCycler] Distance before relax: {dist_pre:.3f} Å")

        pdb_min, ret = runner(
            input_pdb,
            output_pdb,
            return_info=True,
            cyclic_chains=cyclic_chains,
            cyclic_opts=cyclic_opts
        )

        # Post-relax distance
        dist_post = self._calc_atom_distance_from_pdb(
            output_pdb,
            chain_id,
            res_idx_head,
            bond_atom_head,
            res_idx_tail,
            bond_atom_tail
        )

        # Cyclization criterion
        if mode == "head_tail":
            threshold = self.HEAD_TAIL_THRESHOLD
        elif mode == "k_to_de":
            threshold = self.K_TO_DE_THRESHOLD
        elif mode == "cys_to_cys":
            threshold = self.CYS_TO_CYS_THRESHOLD
        else:
            threshold = None

        cyclized = (
            dist_post is not None and
            threshold is not None and
            dist_post <= threshold
        )

        if self.log:
            print(
                f"[AutoCycler] Distance after relax: "
                f"{dist_post:.3f} Å" if dist_post is not None else
                "[AutoCycler] Distance after relax: N/A"
            )

            print(
                f"[AutoCycler] Cyclization: "
                f"{'SUCCESS' if cyclized else 'FAILED'} "
                f"(threshold = {threshold:.3f} Å)"
            )

            print(f"[AutoCycler] Energy: {ret['einit']:.3f} -> {ret['efinal']:.3f} kcal/mol")
            print(f"[AutoCycler] Output saved to {output_pdb}\n")

        return {
            "mode": mode,
            "bond_atoms": (bond_atom_head, bond_atom_tail),
            "distance_pre": dist_pre,
            "distance_post": dist_post,
            "cyclized": cyclized,
            "einit": ret["einit"],
            "efinal": ret["efinal"]
        }


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    minimizer = ForceFieldMinimizerAuto(platform='CUDA') # or 'CPU'
    ret = minimizer(input_file, output_file)
    print(ret)