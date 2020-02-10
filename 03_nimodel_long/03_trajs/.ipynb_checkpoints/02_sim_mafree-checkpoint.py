#!/usr/bin/env python

import pickle
import random

import numpy

nsample = 10000
in_dir_all = '../01_fitness/'
out_dir = 'trajs/'
N_list = (16, 32, 64, 128, 256, 512, 1024)
states = 'AT'
sd = dict(zip(states, range(2)))

def main():
    for N in N_list:
        in_dir = f'{in_dir_all}site_{N:04d}/'
        ivals = numpy.load(f'{in_dir}ivals.npy')
        best_gt = tuple([states[x] for x in ivals.argmax(axis=1)])
        traj_list = []
        traj_fit_list = []
        for i in range(nsample):
            v = list(best_gt)
            path = [''.join(v), ]
            fits = [get_fitness(v, ivals, sd), ]
            for __ in range(100):
                v_next = mutate(v)
                v = v_next
                path.append(''.join(v))
                fits.append(get_fitness(v, ivals, sd))
            traj_list.append(tuple(path))
            traj_fit_list.append(tuple(fits))
            print(f'{N}\ttrajectory {i}')
        with open(f'{out_dir}02_traj_mafree_{N:04d}.pkl',
                  'wb') as f:
            pickle.dump(traj_list, f)
        with open(f'{out_dir}02_trajfit_mafree_{N:04d}.pkl',
                  'wb') as f:
            pickle.dump(traj_fit_list, f)


def get_fitness(v, ivals, sd):
    return numpy.sum([ivals[x, sd[v[x]]] for x in range(ivals.shape[0])])


def mutate(v):
    x = random.choice(range(len(v)))
    if v[x] == 'A':
        v[x] = 'T'
    else:
        v[x] = 'A'
    return v


if __name__ == '__main__':
    main()