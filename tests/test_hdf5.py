import h5py
import argparse
import IPython

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True)
args = parser.parse_args()

with h5py.File(args.file_path, "r") as f:
    print(f.keys())
    IPython.embed()