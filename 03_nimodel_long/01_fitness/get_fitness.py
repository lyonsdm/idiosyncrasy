#!/usr/bin/env python

import itertools
import numpy
import os
import pickle
import subprocess

N_list = (16, 32, 64, 128, 256, 512, 1024)
out_dir = f'./'


def main():
    numpy.random.seed(4032)
    for N in N_list:
        dir_name = f'site_{N:04d}/'
        if not os.path.isdir(dir_name):
            subprocess.call(['mkdir', dir_name])
        ivals = numpy.random.normal(size=(N, 2))
        numpy.save(f'{dir_name}ivals.npy', ivals)


if __name__ == '__main__':
    main()
