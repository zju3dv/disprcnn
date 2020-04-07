import argparse
import os

import torch
from tqdm import tqdm
import pickle


def split(dir_name, split_path):
    os.makedirs(os.path.join(dir_name, 'predictions'), exist_ok=True)
    predictions = torch.load(os.path.join(dir_name, 'predictions.pth'), 'cpu')
    lps, rps = predictions['left'], predictions['right']
    with open(split_path) as f:
        imgids = f.read().splitlines()
    for i in tqdm(range(len(lps))):
        lp, rp = lps[i], rps[i]
        pickle.dump({'left': lp, 'right': rp},
                    open(os.path.join(dir_name, 'predictions', imgids[i] + '.pkl'), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', required=True)
    parser.add_argument('--split_path', required=True)
    args = parser.parse_args()
    dir_name = os.path.dirname(args.prediction_path)
    split_path = args.split_path
    if '%s' in dir_name and '%s' in split_path:
        for s in ('train', 'val'):
            split(dir_name % s, split_path % s)
    else:
        split(dir_name, split_path)


if __name__ == '__main__':
    main()
