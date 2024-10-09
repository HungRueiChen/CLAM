import numpy as np
import json
import pandas as pd
from pathlib import Path
import argparse
import os

# arguments
parser = argparse.ArgumentParser(description="Create csv for CLAM from cohort json file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--json_file", help = 'json file containing hierarchy of splits, classes, then slide ids')
parser.add_argument("--seg_csv", help = 'path to segmentation csv file, named process_list_autogen.csv by default')
parser.add_argument("--dataset_csv_dir", help = 'directory to store dataset csv files')
args = parser.parse_args()

with open(Path(args.json_file), 'r') as f:
    cohort_dict = json.load(f)
df = pd.read_csv(Path(args.seg_csv))
training_df = pd.DataFrame()
seg_dir = Path(args.seg_csv).parent
training_dict = {'case_id': [], 'label': []}
test_df = pd.DataFrame()
test_dict = {'case_id': [], 'label': []}

for subset, class_dict in cohort_dict.items():
	for disease, pid_list in class_dict.items():
		wsi_names = [x + '.mrxs' for x in pid_list]
		temp = df.loc[df['slide_id'].isin(wsi_names)]
		if subset == 'test':
			test_seg_df = pd.concat([test_seg_df, temp])
			test_dict['case_id'] += pid_list
			test_dict['label'] += [disease] * len(pid_list)
		else:
			training_seg_df = pd.concat([training_seg_df, temp])
			training_dict['case_id'] += pid_list
			training_dict['label'] += [disease] * len(pid_list)

# wrap up
os.makedirs(Path(args.dataset_csv_dir), exist_ok = True)
training_dict['slide_id'] = training_dict['case_id']
training_dat_df = pd.DataFrame(training_dict)
training_dat_df.to_csv(Path(args.dataset_csv_dir) / 'training.csv', index = False)
training_seg_df.to_csv(seg_dir / 'training_process_list.csv', index = False)
test_dict['slide_id'] = test_dict['case_id']
test_dat_df = pd.DataFrame(test_dict)
test_dat_df.to_csv(Path(args.dataset_csv_dir) / 'test.csv', index = False)
test_seg_df.to_csv(seg_dir / 'test_process_list.csv', index = False)
