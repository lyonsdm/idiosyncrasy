#!/usr/bin/env python

import os
import pickle
import random
import subprocess

import numpy

in_dir_all = './'
out_dir_all = './'


def main():
    random.seed(4032)
    dir_list = [x for x in os.listdir(in_dir_all)
                if os.path.isdir(f'{in_dir_all}{x}') and x.startswith('site')]
    for dirx in dir_list:
        in_dir = f'{in_dir_all}{dirx}/'
        out_dir = f'{out_dir_all}{dirx}/'
        if not os.path.isdir(out_dir):
            subprocess.call(['mkdir', out_dir])
        ivals = numpy.load(f'{in_dir}ivals.npy')
        N = ivals.shape[0]
        sd = dict(zip('AT', range(2)))
        mut_effects_dict = {}
        for i in range(N):
            mut_tuple = (i, 'A', 'T')
            print(mut_tuple)
            gt_pair_fitlist = sample_gt_pairs(mut_tuple, ivals, sd, 1000)
            mut_effects_dict[mut_tuple] = gt_pair_fitlist
        for i in range(N):
            mut_tuple = (i, 'T', 'A')
            print(mut_tuple)
            gt_pair_fitlist = sample_gt_pairs(mut_tuple, ivals, sd, 1000)
            mut_effects_dict[mut_tuple] = gt_pair_fitlist
        print(len(mut_effects_dict))
        with open(f'{out_dir}mut_effect_dict.pkl', 'wb') as f:
            pickle.dump(mut_effects_dict, f)


def get_fitness(gt, ivals, sd):
    return numpy.sum([ivals[x, sd[gt[x]]] for x in range(ivals.shape[0])])


def sample_gt_pairs(mut_tuple, ivals, sd, nsample):
    sitestate_list_x, sitestate_list_y = [], []
    fitmat_x = numpy.zeros((nsample, ivals.shape[0]))
    fitmat_y = numpy.zeros((nsample, ivals.shape[0]))
    for i in range(ivals.shape[0]):
        if i == mut_tuple[0]:
            sitestate_list_x.append(tuple(mut_tuple[1] * nsample))
            sitestate_list_y.append(tuple(mut_tuple[2] * nsample))
            fitmat_x[:, i] = numpy.ones(nsample) * ivals[i, sd[mut_tuple[1]]]
            fitmat_y[:, i] = numpy.ones(nsample) * ivals[i, sd[mut_tuple[2]]]
        else:
            states = random.choices('AT', k=nsample)
            sitestate_list_x.append(states)
            sitestate_list_y.append(states)
            tmp = [ivals[i, sd[x]] for x in states]
            fitmat_x[:, i] = tmp
            fitmat_y[:, i] = tmp
    fit_list = []
    for i in range(nsample):
        gt_x = ''.join([x[i] for x in sitestate_list_x])
        gt_y = ''.join([x[i] for x in sitestate_list_y])
        fit_x = fitmat_x[i, :].sum()
        fit_y = fitmat_y[i, :].sum()
        fit_list.append((gt_x, gt_y, fit_x, fit_y))
    return fit_list


if __name__ == '__main__':
    main()
