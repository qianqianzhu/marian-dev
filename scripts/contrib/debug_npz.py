#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import numpy as np


DESC = "Debugs .npz file."


def main():
    args = parse_args()
    model = np.load(args.npz)
    for name in model.files:
        obj = model[name]
        print(name, ":", "shape=", obj.shape, "size=", obj.size)
        print(obj)
    model.close()


def parse_args():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("npz", help="path to .npz file")
    return parser.parse_args()


if __name__ == "__main__":
    main()
