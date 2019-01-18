import pandas as pd
import numpy as np
import math
import argparse
import time
import sys

from sklearn.preprocessing import PowerTransformer
# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.svm import LinearSVC, LinearSVR

_random_seed_ = 42
_test_size_ = 0.25


def get_combination_masks(cancer_types, drug_names, drugs_expression_df):
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


def train_test_split(X, y, test_p=0.25, n_iter=10, random_state=42):
    y_test = []
    while len(np.unique(y_test)) < 2 and n_iter > 1:
        n_iter -= 1
        test_size = math.floor(test_p * len(X))
        ids = np.unique(X.index.values)
        test_index = np.random.choice(ids, size=test_size)
        X_train = X[~X.index.isin(test_index)]
        X_test = X[X.index.isin(test_index)]
        y_train = y[~y.index.isin(test_index)]
        y_test = y[y.index.isin(test_index)]
    return X_train, X_test, y_train, y_test

def feature_selection(X_train, y_train, model, test_mask, num_features=300, variance_threshold=0, step=0.05, verbose=0, n_jobs=8):
    # Train test split
    n = len(y)

    # Apply variance threshold on features
    # TODO: note that this is probably removing dummy variables with low variance
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    variance_selector.fit(X_train)
    selected_features = X_train.columns[variance_selector.get_support()]
    X_train_selected = X_train.loc[:, selected_features]

    # Recursive feature elimination with cross validation
    selector = RFECV(model, min_features_to_select=num_features, step=step, verbose=verbose, n_jobs=n_jobs)
    # TODO: figure out way to output scores of intermediate selectors
    X_new = selector.fit_transform(X_train_selected, y_train)
    return selector, X_new.columns


def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    score = model.score(X_test)
    return y_pred, score

def main():
    local_time = time.localtime()
    output_dir_name = "{}_{}_{}".format(local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

    #TODO: move to separate argparse function
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--features', help='path to features tsv. must have column called \'pog_id\'', required=True)
    parser.add_argument('-y', '--labels', help='path to labels tsv. must have columns: \'pog_id\', \'drug_name\', \'response\', \'cancer_cohort\'', required=True)
    parser.add_argument('-o', '--out_dir', help='where to save the output files', default=output_dir_name)
    parser.add_argument('-m', '--model', choices=['svc', 'svr'], help='select the model', required=True)
    parser.add_argument('-v', '--variance_threshold', default=0, help='before RFE, filter features based on variance. Default: 0')    
    parser.add_argument('-s', '--random_seed', default=42, help='random seed')

    args = parser.parse_args()
    expression_file_path = args.features
    drugs_file_path = args.labels
    output_dir = args.out_dir
    variance_threshold = float(args.variance_threshold)
    results_path = output_dir

    _random_seed_ = args.random_seed

    # Use args.model to select model
    if args.model == 'svc':
        discretize = True
        model = LinearSVC(max_iter=50000)
    elif args.model == 'svr':
        discretize = False
        model = LinearSVR(max_iter=50000)

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

    drug_dummies = pd.get_dummies(drugs_expression_df['drug_name'])
    drugs_expression_df = drugs_expression_df.join(drug_dummies)
    X_columns = np.append(expression_df.columns.values, drug_dummies.columns.values)

    cancer_types = np.unique(drugs_expression_df['cancer_cohort'])
    drug_names = np.unique(drugs_expression_df['drug_name'])
    mask, mask_labels = get_combination_masks(cancer_types, drug_names, drugs_expression_df)

    drugs_expression_df = drugs_expression_df.set_index('pog_id')
    report_rows = []
    for mask, mask_label in zip(mask, mask_labels):
        drugs_expression_sel_df = drugs_expression_df[mask]
        cancer_type, drug_name = mask_label

        # Set features (X) and labels (y)
        X = drugs_expression_sel_df.loc[:, X_columns]
        y = drugs_expression_sel_df.loc[:, 'response']
        n = len(y)

        if len(y) < 10:
           # print('Skipping cohort {} and drug name {} with n={}'.format(cancer_type, drug_name, len(y)))
           continue

        # Power transform y
        power_transformer = PowerTransformer(method='box-cox', standardize=True)
        y_trans = power_transformer.fit_transform(y.values.reshape(-1, 1))[:, 0]

        # Determine test set mask
        X_train, X_test, y_train, y_test = train_test_split(X, y_trans)

        if len(np.unique(y_test)) < 2:
            print('Skipping cancer {} and drug {} due to not finding a proper split.'.format(cancer_type, drug_name))
         
        if discretize:
            # Discretize y if binary classification problem
            # TODO: In case not binary classification, implement a better discretization
            y_discrete = y_trans > 0
        selector, selected_columns = feature_selection(X_train, y_train, model, variance_threshold=variance_threshold)

        y_pred, score = test_model(X_test[:, selected_columns], y_test, selector)
        labels = pd.DataFrame({'y': y, 'y_trans': y_trans})
        report_rows.append([cancer_type, drug_name, n, score])
        X_selected = X[:, selected_columns]
        X_selected = X_selected.join(labels)
        X_selected = X_selected.join(y_pred)
        X_selected.to_csv(results_path + '/data/{}_{}.tsv'.format(cancer_type, drug_name), sep='\t')
        # print('Analyzing cancer cohort: {} and drug name: {}'.format(cancer_type, drug_name))

    # pd.DataFrame(list(scores.values()), index=list(scores.keys()), columns=['Score']).sort_values('Score').to_csv('report.tsv', sep='\t')
    pd.DataFrame(report_rows, columns=['cancer_type', 'drug_name', 'n', 'score']).sort_values('score').to_csv(results_path + '/report.tsv', sep='\t')
    with open(results_path + '/output.txt', 'w+') as results_file:
        print('command line arguments (sys.argv object): {}'.format(sys.argv), file=results_file)
        print('expression data path: {}'.format(expression_file_path), file=results_file)
        print('drugs data path: {}'.format(drugs_file_path), file=results_file) 


if __name__ == '__main__':
    main()
