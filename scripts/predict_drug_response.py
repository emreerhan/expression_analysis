_random_seed_ = 42
_test_size_ = 0.25

import pandas as pd
import numpy as np
import math
import argparse

from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.svm import LinearSVC


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')


def RFE_model(X, y, model, test_size=0.25, num_features=1000, variance_threshold=0, step=0.05, verbose=0):
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size_, random_state=_random_seed_)
    n = len(y)

    # Apply variance threshold on features
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    variance_selector.fit(X_train)
    selected_features = X_train.columns[variance_selector.get_support()]
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]

    # Recursive feature elimination
    selector = RFE(model, num_features, step=step, verbose=verbose)
    selector = selector.fit(X_train_selected, y_train)
    score = selector.score(X_test_selected, y_test)
    y_pred = selector.predict(X_test_selected)
    # print('Score: {}'.format(score))
    return([n, score], y_pred)

def get_masks(cancer_types, drug_names, drugs_expression_df):
    masks = []
    mask_labels = []
    for cancer_type in cancer_types:
        masks.append(drugs_expression_df['cancer_cohort'] == cancer_type)
        mask_labels.append((cancer_type, 'All'))
        drug_mask = False
        for drug_name in drug_names:
            if not drug_mask:
                masks.append(drugs_expression_df['drug_name'] == drug_name)
                mask_labels.append(('All', drug_name))
                drug_mask = True
            masks.append((drugs_expression_df['cancer_cohort'] == cancer_type) & (drugs_expression_df['drug_name'] == drug_name))
            mask_labels.append((cancer_type, drug_name))
    return masks, mask_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--features', help='path to features tsv. must have column called \'pog_id\'', required=True)
    parser.add_argument('-y', '--labels', help='path to labels tsv. must have columns: \'pog_id\', \'drug_name\', \'response\', \'cancer_cohort\'', required=True)
    parser.add_argument('-o', '--out_dir', help='where to save the output files', required=True)
    parser.add_argument('-m', '--model', choices=['svc'], help='select the model', required=True)
    parser.add_argument('-v', '--variance_threshold', default=0, help='before RFE, filter features based on variance. Default: do not filter')    
    parser.add_argument('-d', '--discretize', action='store_true', help='if classification, use box-cox transform and discretize response by < 0 or >= 0')
    parser.add_argument('-s', '--random_seed', default=42, help='random seed')

    args = parser.parse_args()
    expression_file_path = args.features
    drugs_file_path = args.labels
    output_dir = args.out_dir
    discretize = args.discretize
    variance_threshold = args.variance_threshold
    results_path = output_dir

    _random_seed_ = args.random_seed

    # TODO: Use args.model to select model
    model = LinearSVC(max_iter=50000)

    # TODO: Use test_size param
    # TODO: Determine transform on y, for now assume box-cox

    expression_df = pd.read_csv(expression_file_path, sep='\t')
    expression_df = expression_df.set_index('pog_id')

    drugs_df = pd.read_csv(drugs_file_path, sep='\t')
    drugs_selected_df = drugs_df[['pog_id', 'drug_name', 'response', 'cancer_cohort']]

    # # Prepare features and labels

    # ## Join drugs and expression tables

    drugs_expression_df = drugs_selected_df.join(expression_df, on='pog_id', how='inner')
    drugs_expression_df = drugs_expression_df.drop_duplicates()

    cancer_types = np.unique(drugs_expression_df['cancer_cohort'])
    drug_names = np.unique(drugs_expression_df['drug_name'])
    mask, mask_labels = get_masks(cancer_types, drug_names, drugs_expression_df)

    rows = []
    for mask, mask_label in zip(mask, mask_labels):
        drugs_expression_sel_df = drugs_expression_df[mask]
        cancer_type, drug_name = mask_label

        # Set features (X) and labels (y)
        X = drugs_expression_sel_df.loc[:, expression_df.columns]
        y = drugs_expression_sel_df.loc[:, 'response']
        n = len(y)

        if len(y) < 10:
            # print('Skipping cohort {} and drug name {} with n={}'.format(cancer_type, drug_name, len(y)))
            continue

        if discretize:
            # Power transform y
            power_transformer = PowerTransformer(method='box-cox', standardize=True)
            y_trans = power_transformer.fit_transform(y.values.reshape(-1, 1))[:, 0]
            # Discretize y
            y_discrete = y_trans > 0
            new_row, y_pred = RFE_model(X, y_discrete, model, variance_threshold=variance_threshold)
            labels = pd.DataFrame({'y': y, 'y_trans': y_trans})
        else:
            new_row, y_pred = RFE_model(X, y, model)
            labels = y
        rows.append([cancer_type, drug_name] + new_row)
        X.join(labels).to_csv(results_path + '/{}_{}.tsv'.format(cancer_type, drug_name), sep='\t')
        # print('Analyzing cancer cohort: {} and drug name: {}'.format(cancer_type, drug_name))

    # pd.DataFrame(list(scores.values()), index=list(scores.keys()), columns=['Score']).sort_values('Score').to_csv('report.tsv', sep='\t')
    pd.DataFrame(rows, columns=['cancer_type', 'drug_name', 'n', 'score']).sort_values('score').to_csv(results_path + '/report.tsv', sep='\t')
    print('expression data path: {}'.format(expression_file_path), file=open(results_path + '/output.txt', 'w+'))
    print('drugs data path: {}'.format(drugs_file_path), file=open(results_path + '/output.txt', 'w+')) 


if __name__ == '__main__':
    main()