import numpy as np
import time, os, argparse
from cmtip.prep_data import *
from cmtip.prep_data import load_checkpoint, save_checkpoint, load_h5
import h5py, os, pickle, sys
import numpy as np

"""
Convert checkpoint file from cmtip to spinifel format or vice versa.
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description="Checkpoint file converter between spinifel and cmtip formats.")
    parser.add_argument('-c', '--cpt', help='Checkpoint file in h5/cmtip or pickle/spinifel format.')
    parser.add_argument('-o', '--output', help='Output directory for conerted checkpoint file', default='.')
    parser.add_argument('-r', '--rank', help='Rank of data to load; use -1 if sequential', type=int, default=0)

    return vars(parser.parse_args())

def main():

    args = parse_input()
    if args['cpt'].split(".")[-1] == 'h5':
        print("Converting from cmtip to spinifel format.")

        # load checkpoint file
        checkpoint = load_checkpoint(input_file=args['cpt'], rank=args['rank'])
        if checkpoint['generation'] != 0:
            key_name = [key for key in checkpoint.keys() if key.startswith('orientations')][0]
            checkpoint['orientations'] = checkpoint.pop(key_name)

        # save in pickle format
        fname = os.path.join(args['output'], f"generation_{checkpoint['generation']}.pickle")
        with open(fname, 'wb') as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved file to {fname}")

    elif args['cpt'].split(".")[-1] == 'pickle':
        print("Converting from spinifel to cmtip format.")

        # load checkpoint file and rename keys as needed
        with open(args['cpt'], 'rb') as handle:
            checkpoint = pickle.load(handle)
        if rank != -1:
            checkpoint["orientations_r{args['rank']}"] = checkpoint.pop('orientations')

        # convert to h5 format
        checkpoint['generation'] = int(args['cpt'].split("generation_")[1].split(".")[0])
        save_checkpoint(checkpoint['generation'], args['output'], checkpoint)

    else:
        print("Invalid checkpoint file")
        sys.exit()


if __name__ == '__main__':
    main()
