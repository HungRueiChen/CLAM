import numpy as np
import csv
import random
import cv2
import os
from pathlib import Path
import pandas as pd
import itertools
import argparse
from shutil import rmtree, copy
import json
import roc_utils as ru
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# arguments
parser = argparse.ArgumentParser(description="Test the Best Model on Given Directory with Metrics ", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("csv_path", help = 'foldX.csv under RESULT_DIR')
args = parser.parse_args()

# json encoder for special data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# define variables
classes = ['ALL', 'AML', 'CML', 'Lymphoma', 'MM']
log_dir = Path(args.csv_path).parent

# Test Section
# Define functions
def calculate_metrics(y_true, y_pred, f_name, result_dict, classes):
    # initialize
    cm_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    
    # accuracy = micro f1
    correct = np.equal(y_true, y_pred)
    acc = sum(correct) / len(correct)
    
    # balanced accuracy
    recalls = [cm_norm[x][x] for x in range(len(classes))]
    bal_acc = np.mean(recalls)
    
    # f1 score (macro/micro)
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    f1_micro = f1_score(y_true, y_pred, average = 'micro') # should = accuracy

    # Matthews correlation coefficient
    matthews = matthews_corrcoef(y_true, y_pred)

    # Cohen's kappa
    cohen = cohen_kappa_score(y_true, y_pred)
    
    # save to dict
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['accuracy'] = acc
    result_dict[f_name]['balanced_accuracy'] = bal_acc
    result_dict[f_name]['f1_macro'] = f1_macro
    result_dict[f_name]['matthews_CC'] = matthews
    result_dict[f_name]['cohen_kappa'] = cohen

def draw_cm(y_true, y_pred, log_dir, f_name, result_dict, classes):
    # initialize
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize = 'true')
    
    fig_cm = plt.figure(figsize = (18, 6))

    for i, mtx in enumerate([cm, cm_norm.round(3)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(mtx, interpolation = 'nearest', vmin = 0, cmap = plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 0, fontsize = 10)
        plt.yticks(tick_marks, classes, fontsize = 10)
        thresh = mtx.max() / 2.
        for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
            plt.text(j, i, mtx[i, j], horizontalalignment = "center", fontsize = 15, color = "white" if mtx[i, j] > thresh else "black")

        plt.ylabel('Groundtruth', fontsize = 12)
        plt.xlabel('Prediction', fontsize = 12)
        subtitle = 'Normalized Confusion Matrix' if i else 'Confusion Matrix'

    fig_cm.savefig(log_dir / f'{f_name}_confusion_matrix.png', dpi = fig_cm.dpi, bbox_inches = 'tight')
    plt.tight_layout()
    plt.close()
    
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['confusion_matrix'] = cm
    result_dict[f_name]['normalized_cm'] = cm_norm
    
def draw_roc_curve(y_true, y_pred, log_dir, f_name, result_dict, classes):
    roc_data = {}

    fig_roc = plt.figure(figsize = (18, 5.6))
    plt.subplot(1, 3, 1)
    colors = ['firebrick', 'orange', 'limegreen', 'deepskyblue', 'deeppink']

    # roc curve as per class
    for i, study_group in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        area = auc(fpr, tpr)
        roc_data[study_group] = {'fpr': fpr, 'tpr': tpr, 'auc': area}

        plt.plot(fpr, tpr, label = f'{study_group}, area = {area:.3f}', color = colors[i], linewidth = 1)

    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.legend(loc = "lower right", fontsize = 8)

    # roc curves micro
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    area = auc(fpr, tpr)
    roc_data['micro'] = {'fpr': fpr, 'tpr': tpr, 'auc': area}

    # roc curves macro
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    mean_tpr = np.zeros_like(fpr_grid)
    for study_group in classes:
        mean_tpr += np.interp(fpr_grid, roc_data[study_group]['fpr'], roc_data[study_group]['tpr'])
    # average it and compute AUC
    mean_tpr /= len(classes)
    mac_area = auc(fpr_grid, mean_tpr)
    fpr_grid = np.insert(fpr_grid, 0, 0)
    mean_tpr = np.insert(mean_tpr, 0, 0)
    roc_data['macro'] = {'fpr': fpr_grid, 'tpr': mean_tpr, 'auc': mac_area}

    # plot roc micro and macro
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label = f'micro-average, area = {area:.3f}', color = 'red', linewidth = 1)
    plt.plot(fpr_grid, mean_tpr, label = f'macro-average, area = {mac_area:.3f}', color = 'blue', linewidth = 1)
    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.legend(loc = "lower right", fontsize = 8)
    
    # bootstrap TI of roc micro
    n_samples = 10000 if 'Patient' in f_name else 1000
    ret_mean = ru.compute_roc_bootstrap(X = y_pred.ravel(), y = y_true.ravel(), pos_label = 1, n_bootstrap = n_samples, return_mean = True)
    tpr_sort = np.sort(ret_mean.tpr_all, axis=0)
    tpr_lower = tpr_sort[int(0.025 * n_samples), :]
    tpr_upper = tpr_sort[int(0.975 * n_samples), :]
    roc_data['micro']['auc_mean'] = ret_mean["auc_mean"]
    roc_data['micro']['auc95_ci'] = ret_mean["auc95_ci"][0]
    roc_data['micro']['auc95_ti'] = ret_mean["auc95_ti"][0]
    roc_data['micro']['auc_std'] = ret_mean["auc_std"]
    
    # plot roc micro with TI
    plt.subplot(1, 3, 3)    
    plt.plot(fpr, tpr, label = f'micro-average, area = {area:.3f}', color = 'red', linewidth = 1, zorder = 3)
    plt.fill_between(ret_mean.fpr, tpr_lower, tpr_upper, color='gray', alpha=.3, label=f'95% interval, area = {ret_mean.auc95_ti[0]}', zorder=2)
    plt.plot([0, 1], [0, 1], 'k--', lw = 1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.legend(loc = "lower right", fontsize = 8)

    fig_roc.savefig(log_dir / f'{f_name}_ROC_curve.png', dpi = fig_roc.dpi, bbox_inches = 'tight')
    plt.close()
    
    if f_name not in result_dict:
        result_dict[f_name] = {}
    result_dict[f_name]['roc_data'] = roc_data


# load pred scores
df = pd.read_csv(args.csv_path, names=['slide_id','Y','Y_hat','p0','p1','p2','p3','p4'], header=0)
y_true = df['Y'].to_numpy(dtype = int)
y_true_onehot = np.zeros((len(y_true), 5), dtype = float)
for idx, y in enumerate(y_true):
    y_true_onehot[idx][y] = 1.0

y_pred = df['Y_hat'].to_numpy(dtype = int)
y_pred_probs = df[['p0','p1','p2','p3','p4']].to_numpy(dtype = float)
    
# calculate metrics
result_dict = {}

# patient level, mode count for roc curve
f_name = 'CLAM_1fold_default_params'
calculate_metrics(y_true, y_pred, f_name, result_dict, classes)
draw_cm(y_true, y_pred, log_dir, f_name, result_dict, classes)
draw_roc_curve(y_true_onehot, y_pred_probs, log_dir, f_name, result_dict, classes)

with open(log_dir / 'test_metrics.json', 'w') as f:
    json.dump(result_dict, f, cls = NumpyEncoder)

method_list = ['CLAM_1fold_default_params']
metric_list = ['accuracy', 'balanced_accuracy', 'f1_macro', 'matthews_CC', 'cohen_kappa', 'auc_micro', 'auc_macro']
result_arr = np.zeros((len(method_list), len(metric_list)))
for i, method in enumerate(method_list):
    for j, metric in enumerate(metric_list):
        if metric == 'auc_micro':
            result_arr[i][j] = result_dict[method]['roc_data']['micro']['auc']
        elif metric == 'auc_macro':
            result_arr[i][j] = result_dict[method]['roc_data']['macro']['auc']
        else:
            result_arr[i][j] = result_dict[method][metric]
result_df = pd.DataFrame(result_arr, columns = metric_list, index = method_list)
result_df.to_csv(log_dir / 'test_metrics.csv')
