#!/usr/bin/env python

import pandas
import pickle
import subprocess

fit_data_file = '../00_data/ExpData.txt'
out_dir = './'
wildtype = open(fit_data_file).readlines()[1].split()[3]


def main():
    fit_df = pandas.read_csv(fit_data_file, sep='\t')
    gt_mut_list = []
    gt_fit_list = []
    gt_fitrep_list = []
    mut_all_set = set([])
    gt_dict = {}
    for i in range(fit_df.shape[0]):
        if i == 0:
            mut_set = set([])
        else:
            posl = [int(x) for x in fit_df['Pos'][i].split()]
            nucl = [x for x in fit_df['Nuc'][i].split()]
            mut_set = set([tuple(x) for x in zip(posl, nucl)])
        gt_mut_list.append(mut_set)
        gt_fit_list.append(fit_df['Fit'][i] / 0.5)
        gt_fitrep_list.append((
            fit_df['FitS1'][i] / 0.5,
            fit_df['FitS2'][i] / 0.5,
            fit_df['FitS3'][i] / 0.5,
            fit_df['FitS4'][i] / 0.5,
            fit_df['FitS5'][i] / 0.5,
            fit_df['FitS6'][i] / 0.5,
        ))
        mut_all_set.update(mut_set)
        gt_dict[frozenset(mut_set)] = i
    print(f'# genotypes: {len(gt_mut_list)}, # mutations: {len(mut_all_set)}')
    with open(f'{out_dir}gt_mut_list.pkl', 'wb') as f:
        pickle.dump(gt_mut_list, f)
    with open(f'{out_dir}gt_fit_list.pkl', 'wb') as f:
        pickle.dump(gt_fit_list, f)
    with open(f'{out_dir}gt_fitrep_list.pkl', 'wb') as f:
        pickle.dump(gt_fitrep_list, f)
    with open(f'{out_dir}gt_dict.pkl', 'wb') as f:
        pickle.dump(gt_dict, f)


if __name__ == '__main__':
    main()