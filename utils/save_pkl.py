import os
import pickle
from tqdm import tqdm

from data.pdb_reader import load_pdb_as_dict

def pdb_to_pickle(pdb_folder, out_folder):
    """
    将 pdb_folder 中所有 .pdb 文件读取内容并 pickle 到 out_folder/*.pkl
    """
    os.makedirs(out_folder, exist_ok=True)

    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]

    for pdb_file in tqdm(pdb_files, desc="Pickling PDB files"):
        pdb_path = os.path.join(pdb_folder, pdb_file)
        pkl_path = os.path.join(out_folder, pdb_file.replace(".pdb", ".pkl"))

        # 读取 PDB 内容
        pdb_content = load_pdb_as_dict(pdb_path)

        # 保存到 pickle
        with open(pkl_path, "wb") as f:
            pickle.dump(pdb_content, f)

if __name__ == "__main__":
    pdb_folder = "inference_output"
    out_folder = "./data/pkl/c_linear_from_linear/CPSea"

    pdb_to_pickle(pdb_folder, out_folder)
