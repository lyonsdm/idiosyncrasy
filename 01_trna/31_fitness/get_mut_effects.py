#!/usr/bin/env python

import pandas
import pickle
import subprocess

fit_data_file = '../00_data/ExpData.txt'
in_dir = './'
out_dir = './'
wildtype = open(fit_data_file).readlines()[1].split()[3]


def main():
    gt_mut_list = pickle.load(open(f'{in_dir}/gt_mut_list.pkl', 'rb'))
    gt_fit_list = pickle.load(open(f'{in_dir}/gt_fit_list.pkl', 'rb'))
    gt_fitrep_list = pickle.load(open(f'{in_dir}/gt_fitrep_list.pkl', 'rb'))
    gt_graph = pickle.load(open(f'{in_dir}/gt_graph.pkl', 'rb'))
    mut_effects_dict = {}
    for j in range(len(gt_mut_list)):
        cnt = 0
        neighbors = gt_graph[j]
        # get mutations for one genotype
        for neighbor_idx, mut_tuple in neighbors:
            if mut_tuple not in mut_effects_dict:
                mut_effects_dict[mut_tuple] = []
            mut_effects_dict[mut_tuple].append(
                (j, neighbor_idx, gt_fit_list[j], gt_fit_list[neighbor_idx],
                 gt_fitrep_list[j], gt_fitrep_list[neighbor_idx])
            )
            cnt += 1
        print(f'{j}-th genotype with {len(gt_mut_list[j])} mutations; '
              f'{cnt} mutation effects found based on it')
    with open(f'{out_dir}mut_effect_dict.pkl', 'wb') as f:
        pickle.dump(mut_effects_dict, f)


if __name__ == '__main__':
    main()