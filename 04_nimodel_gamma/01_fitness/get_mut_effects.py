#!/usr/bin/env python

import os
import pickle
import subprocess

in_dir_all = './'
out_dir_all = './'


def main():
    dir_list = [x for x in os.listdir(in_dir_all)
            if os.path.isdir(f'{in_dir_all}{x}') and x.startswith('nlevel')]
    for dirx in dir_list:
        in_dir = f'{in_dir_all}{dirx}/'
        out_dir = f'{out_dir_all}{dirx}/'
        if not os.path.isdir(out_dir):
            subprocess.call(['mkdir', out_dir])
        gt_mut_list = pickle.load(open(f'{in_dir}/gt_mut_list.pkl', 'rb'))
        gt_fit_list = pickle.load(open(f'{in_dir}/gt_logfit_list.pkl', 'rb'))
        gt_graph = pickle.load(open(f'{in_dir}/gt_graph.pkl', 'rb'))
        mut_effects_dict = {}
        for j in range(len(gt_mut_list)):
            cnt = 0
            neighbors = gt_graph[j]
            for neighbor_idx, mut_tuple in neighbors:
                if mut_tuple not in mut_effects_dict:
                    mut_effects_dict[mut_tuple] = []
                mut_effects_dict[mut_tuple].append(
                    (j, neighbor_idx, gt_fit_list[j], gt_fit_list[neighbor_idx])
                )
                cnt += 1
            print(f'{j}-th genotype with {len(gt_mut_list[j])} mutations; '
                  f'{cnt} mutation effects found based on it')
        print(len(mut_effects_dict))
        with open(f'{out_dir}mut_effect_dict.pkl', 'wb') as f:
            pickle.dump(mut_effects_dict, f)


if __name__ == '__main__':
    main()