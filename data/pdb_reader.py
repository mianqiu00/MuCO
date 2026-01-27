import os
import numpy as np
import pickle
import csv
from tqdm import tqdm


atom_types = [
    "N","CA","C","CB","O","CG","CG1","CG2","OG","OG1","SG","CD","CD1","CD2",
    "ND1","ND2","OD1","OD2","SD","CE","CE1","CE2","CE3","NE","NE1","NE2",
    "OE1","OE2","CH2","NH1","NH2","OH","CZ","CZ2","CZ3","NZ","OXT"
]
atom_order = {a: i for i, a in enumerate(atom_types)}
atom_type_num = 37

restypes = [
    "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"
]
restype_order = {aa: i for i, aa in enumerate(restypes)}

three_to_one = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# parse PDB
def load_pdb_as_dict(pdb_path, default_b=100.0):
    residues = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            chain_id  = line[21].strip()
            resid     = int(line[22:26])

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            try:
                b_factor = float(line[60:66])
            except:
                b_factor = default_b

            # new residue?
            if len(residues) == 0 or resid != residues[-1]["resid"] or chain_id != residues[-1]["chain"]:
                residues.append({
                    "resname": resname,
                    "resid": resid,
                    "chain": chain_id,
                    "atoms": {}
                })

            residues[-1]["atoms"][atom_name] = (x, y, z, b_factor)

    L = len(residues)

    atom_positions = np.zeros((L, atom_type_num, 3), dtype=np.float32)
    atom_mask      = np.zeros((L, atom_type_num), dtype=np.float32)
    b_factors      = np.zeros((L, atom_type_num), dtype=np.float32)

    aatype         = np.zeros((L,), dtype=np.int32)
    residue_index  = np.zeros((L,), dtype=np.int32)
    chain_index    = np.zeros((L,), dtype=np.int32)

    bb_positions   = np.zeros((L, 3), dtype=np.float32)
    bb_mask        = np.zeros((L,), dtype=np.float32)

    chain_to_int = {}
    chain_counter = 0

    # fill arrays
    for i, res in enumerate(residues):
        res3 = res["resname"]
        res1 = three_to_one.get(res3, None)

        # aatype
        if res1 in restype_order:
            aatype[i] = restype_order[res1]
        else:
            aatype[i] = -1

        residue_index[i] = res["resid"]

        if res["chain"] not in chain_to_int:
            chain_to_int[res["chain"]] = chain_counter
            chain_counter += 1
        chain_index[i] = chain_to_int[res["chain"]]

        # atoms
        for atom_name, (x, y, z, b) in res["atoms"].items():
            if atom_name in atom_order:
                a_id = atom_order[atom_name]
                atom_positions[i, a_id] = [x, y, z]
                atom_mask[i, a_id] = 1.0
                b_factors[i, a_id] = b

        # backbone (use CA)
        if "CA" in res["atoms"]:
            bb_mask[i] = 1.0
            bb_positions[i] = res["atoms"]["CA"][:3]

    out = {
        "atom_positions": atom_positions,
        "aatype": aatype,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "b_factors": b_factors,
        "bb_mask": bb_mask,
        "bb_positions": bb_positions,
    }

    return out


def process_all_pdbs(pdb_root="./pdb", out_root="./pkl", csv_path="dataset.csv"):
    rows = []

    cluster_dirs = []
    for root, dirs, files in os.walk(pdb_root):
        if any(f.endswith(".pdb") for f in files):
            cluster_dirs.append(root)

    for cluster_dir in cluster_dirs:
        # cluster_dir = ./pdb/cyclic/CPBind
        if not "align" in cluster_dir:
            continue
        rel_path = os.path.relpath(cluster_dir, pdb_root)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            cluster_type, cluster_name = parts[0], parts[1]
        elif len(parts) == 1:
            cluster_type, cluster_name = "", parts[0]
        else:
            cluster_type, cluster_name = "", ""

        files = [f for f in os.listdir(cluster_dir) if f.endswith(".pdb")]

        print(f"\nProcessing cluster: {cluster_name}")
        for fname in tqdm(files, desc=f"{cluster_name}", ncols=80):
            pdb_path = os.path.abspath(os.path.join(cluster_dir, fname))
            pdb_name = os.path.splitext(fname)[0]

            out_dir = os.path.join(out_root, cluster_type, cluster_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.abspath(os.path.join(out_dir, pdb_name + ".pkl"))

            # PDB → dict → pkl
            data = load_pdb_as_dict(pdb_path)
            with open(out_path, "wb") as f:
                pickle.dump(data, f)

            L = len(data["aatype"])

            rows.append({
                "pdb_path": pdb_path,
                "processed_path": out_path,
                "pdb_name": pdb_name,
                "modeled_seq_len": L,
                "oligomeric_detail": "monomer",
                "helix_percent": 0,
                "coil_percent": 0,
                "strand_percent": 0,
                "radius_gyration": 0,
                "type": cluster_type,
                "cluster": cluster_name, 
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_path", "processed_path", "pdb_name",
                "modeled_seq_len", "oligomeric_detail",
                "helix_percent", "coil_percent", "strand_percent",
                "radius_gyration", "type", "cluster"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("CSV saved:", csv_path)


if __name__ == "__main__":
    process_all_pdbs("./pdb", "./pkl", "metadata.csv")
