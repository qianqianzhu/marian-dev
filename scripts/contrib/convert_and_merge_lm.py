#!/usr/bin/env python3
'''This script is used to produce a new npz file with renamed matrices
 consistent with what the marian interpolated lm expects'''
from sys import argv
from distutils.util import strtobool
import numpy

def outputNPZ(ouput_dict, output_file):
    numpy.savez(output_file, **ouput_dict)

def convertNPZ(input_file):
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
    return new_dict

def mergeNPZ(input_NMT_dict1, input_LM_dict2, convert=True):
    """Merge an LM and a TM"""
    input_NMT_dict1 = dict(numpy.load(input_NMT_dict1))
    if convert:
        input_LM_dict2 = convertNPZ(input_LM_dict2)
    else:
        input_LM_dict2 = dict(numpy.load(input_LM_dict2))

    new_dict = {}
    for key in input_NMT_dict1.keys():
        new_dict[key] = input_NMT_dict1[key]
    for key in input_LM_dict2.keys():
        new_dict[key] = input_LM_dict2[key]
    return new_dict




if __name__ == '__main__':
    if len(argv) != 3 and len(argv) != 4 and len(argv) != 5:
        print("Usage: " + argv[0] + " input.npz output.npz #To convert npz to an LM_NPZ")
        print("Usage: " + argv[0] + " input.NMT.npz input.LM.npz output.npz [true] #Take "\
         "a NMT and LM files and merge them. if the last parameter is set to false, the lm "\
         "file wouldn't be converted first.")
        exit()
    elif len(argv) == 3:
        new_dict = convertNPZ(argv[1])
        outputNPZ(new_dict, argv[2])
    elif len(argv) == 4:
        new_dict = mergeNPZ(argv[1], argv[2])
        outputNPZ(new_dict, argv[3])
    else:
        new_dict = mergeNPZ(argv[1], argv[2], strtobool(argv[4]))
        outputNPZ(new_dict, argv[3])
