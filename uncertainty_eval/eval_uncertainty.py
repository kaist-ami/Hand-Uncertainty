from __future__ import print_function, unicode_literals

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pip
import argparse
import base64
import json
from scipy import stats
from scipy.integrate import simpson

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])


try:
    from scipy.linalg import orthogonal_procrustes
except:
    install('scipy')
    from scipy.linalg import orthogonal_procrustes

from utils.fh_utils import *
from utils.eval_util import EvalUtil, EvalUtilUncertainty
import matplotlib.pyplot as plt

def _search_pred_file(pred_path, pred_file_name):
    """ Tries to select the prediction file. Useful, in case people deviate from the canonical prediction file name. """
    pred_file = os.path.join(pred_path, pred_file_name)
    if os.path.exists(pred_file):
        # if the given prediction file exists we are happy
        return pred_file

    print('Predition file "%s" was NOT found' % pred_file_name)

    # search for a file to use
    print('Trying to locate the prediction file automatically ...')
    files = [os.path.join(pred_path, x) for x in os.listdir(pred_path) if x.endswith('.json')]
    if len(files) == 1:
        pred_file_name = files[0]
        print('Found file "%s"' % pred_file_name)
        return pred_file_name
    else:
        print('Found %d candidate files for evaluation' % len(files))
        raise Exception('Giving up, because its not clear which file to evaluate.')


def main(gt_path, pred_path, output_dir, dataset, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'

    os.makedirs(output_dir, exist_ok=True)

    # load eval annotations
    xyz_list, verts_list = json_load(os.path.join(gt_path, '%s_xyz.json' % set_name)), json_load(
        os.path.join(gt_path, '%s_verts.json' % set_name))

    # load predicted values
    pred_file = _search_pred_file(pred_path, pred_file_name)
    print('Loading predictions from %s' % pred_file)
    with open(pred_file, 'r') as fi:
        pred = json.load(fi)

    uncertainty_file = os.path.join(pred_path, f'{dataset}-val_uncertainty.json')
    print('Loading uncertainty predictions from %s' % uncertainty_file)
    with open(uncertainty_file, 'r') as fii:
        uncertainty = json.load(fii)

    assert len(pred) == 2, 'Expected format mismatch.'
    assert len(pred[0]) == len(xyz_list), 'Expected format mismatch.'
    assert len(pred[1]) == len(xyz_list), 'Expected format mismatch.'
    assert len(pred[0]) == db_size(set_name, dataset)

    # init eval utils
    eval_xyz, eval_uncertainty = EvalUtil(), EvalUtilUncertainty()

    try:
        from tqdm import tqdm
        rng = tqdm(range(db_size(set_name, dataset)))
    except:
        rng = range(db_size(set_name, dataset))

    # iterate over the dataset once
    for idx in rng:
        if idx >= db_size(set_name, dataset):
            break

        xyz, verts = xyz_list[idx], verts_list[idx]
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred, uncertainty_pred = pred[0][idx], uncertainty[0][idx]
        xyz_pred, uncertainty_pred = [np.array(x) for x in [xyz_pred, uncertainty_pred]]

        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )
        eval_uncertainty.feed(np.ones_like(uncertainty_pred), uncertainty_pred)

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    pearson_correlation, error_list, uncertainty_list  = eval_uncertainty.get_measures(eval_xyz.data)

    print(f'Pearson Correlation: {pearson_correlation.statistic}')

    q_1 = np.percentile(uncertainty_list, 1)
    q_99 = np.percentile(uncertainty_list, 99)
    uncertainty_list_1_99 = np.clip(uncertainty_list, a_min=q_1, a_max=q_99)
    pearson_correlation_1_99 = stats.pearsonr(error_list, uncertainty_list_1_99)
    print(f'Pearson Correlation_1_99: {pearson_correlation_1_99.statistic}')

    q_5 = np.percentile(uncertainty_list, 5)
    q_95 = np.percentile(uncertainty_list, 95)
    uncertainty_list_5_95 = np.clip(uncertainty_list, a_min=q_5, a_max=q_95)
    pearson_correlation_5_95 = stats.pearsonr(error_list, uncertainty_list_5_95)
    print(f'Pearson Correlation_5_95: {pearson_correlation_5_95.statistic}')

    error_list = eval_xyz.data
    uncertainty_list = eval_uncertainty.data

    error_np = np.array(error_list).reshape(-1)
    error_np = error_np * 10
    uncertainty_np = np.array(uncertainty_list).reshape(-1)

    uncertainty_sort_idx = np.argsort(uncertainty_np)
    error_sort_idx = np.argsort(error_np)
    random_sort_idx = np.random.permutation(len(error_np))

    error_sorted_np = error_np[uncertainty_sort_idx]
    oracle_np = error_np[error_sort_idx]
    random_np = error_np[random_sort_idx]
    data_num = error_sorted_np.shape[0]
    error_percentage = []
    oracle = []
    error_percentage_oracle = []
    random_sorted = []
    for percentage in [i for i in range(2, 101, 2)]:
        partial_num = int(data_num * (percentage / 100))
        error_percentage.append(error_sorted_np[:partial_num].mean())
        oracle.append(oracle_np[:partial_num].mean())
        error_percentage_oracle.append(error_sorted_np[:partial_num].mean() - oracle_np[:partial_num].mean())
        random_sorted.append(random_np[:partial_num].mean())

    area = simpson(error_percentage, x=[i for i in range(2, 101, 2)])
    area_minus_oracle = simpson(error_percentage_oracle, x=[i for i in range(2, 101, 2)])
    print(f'AUSC: {area}')
    print(f'AUSE: {area_minus_oracle}')

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        fo.write(f'Pearson Correlation: {pearson_correlation.statistic}\n')
        fo.write(f'Pearson Correlation_1_99: {pearson_correlation_1_99.statistic}\n')
        fo.write(f'Pearson Correlation_5_95: {pearson_correlation_5_95.statistic}\n')
        fo.write(f'AUSC: {area}\n')
        fo.write(f'AUSE: {area_minus_oracle}\n')
    print('Scores written to: %s' % score_path)

    print('Evaluation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--pred_file_name', type=str, default='freihand-val.json',
                        help='Name of the eval file.')
    parser.add_argument('--dataset', type=str, choices=['ho3d', 'freihand'],
                        help='Dataset', default='ho3d')
    parser.add_argument('--pred_file_dir', type=str, default='/home/chaeyeon/hand/hand_uncertainty/results',
                        help='Path to the prediction directory.')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='Directory to save results.')
    parser.add_argument('--exp', type=str, default='hamer_ours',
                        help='Name of experiment')
    args = parser.parse_args()

    base = args.exp
    save_dir = os.path.join(args.save_dir, args.dataset)
    # call eval
    main(
        f'{args.dataset}/gt',
        os.path.join(args.pred_file_dir, base),
        os.path.join(save_dir, base),
        pred_file_name=f'{args.dataset}-val.json',
        dataset=args.dataset,
        set_name='evaluation',
    )
