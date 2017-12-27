#!/usr/bin/env python3
'''This script is used to produce a new npz file with renamed matrices
 consistent with what the marian interpolated lm expects'''
from sys import argv
import numpy

def convertNPZ(input_file, output_file):
    '''Converts all the decoder* matrices in the NPZ file to decoder_lm*'''
    input_dict = dict(numpy.load(input_file))
    new_dict = dict()
    # Produce new keys
    for key in input_dict.keys():
        if 'decoder' not in key:
            new_dict[key] = input_dict[key]
        else:
            new_key = key.replace('decoder', 'decoder_lm')
            new_dict[new_key] = input_dict[key]

    # Save back to the array
    numpy.savez(output_file, **new_dict)


if __name__ == '__main__':
    if len(argv) != 3:
        print("Usage: " + argv[0] + " input.npz output.npz")
        exit()
    else:
        convertNPZ(argv[1], argv[2])
