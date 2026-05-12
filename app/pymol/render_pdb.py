import sys
from pathlib import Path

import pymol


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: render_pdb.py input.pdb output.png")
    pdb_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    template = Path(__file__).with_name("all_atom.pml")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pymol.finish_launching(["pymol", "-cq"])
    cmd = pymol.cmd
    cmd.load(str(pdb_path), "mol")
    cmd.do(f"@{template}")
    cmd.set("ray_opaque_background", 0)
    cmd.png(str(out_path), width=900, height=700, dpi=180, ray=1)
    cmd.quit()


if __name__ == "__main__":
    main()
