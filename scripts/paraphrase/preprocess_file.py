#!/usr/bin/env python3
'''This file takes input file and vocabulary file and produces vocabID-d file'''
from sys import argv
from typing import Dict
def getVocabDict(filename: str) -> Dict[str, str]:
    """Reads the yaml file and makes a dictionary out of it"""
    ret = {}
    with open(filename, 'r') as infile:
        for line in infile:
            tokenized = line.strip().split(":")
            txt = None
            num = None
            if len(tokenized) == 2:
                txt = tokenized[0]
                num = tokenized[-1]
            else:
                txt = ":"
                num = tokenized[-1]
            ret[txt] = num
    return ret

if __name__ == '__main__':
    if len(argv) != 3:
        print("Usage: ", argv[0], " yaml_file corpora")
        exit(1)

    vocab = getVocabDict(argv[1])
    with open(argv[2], 'r') as infile2:
        for newline in infile2:
            items = newline.strip().split()
            toprint = []
            for item in items:
                if item in vocab:
                    toprint.append(vocab[item])
                else:
                    toprint.append(vocab["<unk>"])
            print(" ".join(toprint))
