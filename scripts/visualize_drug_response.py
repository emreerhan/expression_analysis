import pandas as pd
import numpy as np
import time
import argparse

from matplotlib import pyplot as plt

# Modified from yoavram's post at https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot
def rand_jitter(arr):
    jitter = 0.0001
    return arr + np.random.randn(len(arr)) * jitter

def discrete_plot(y_trans, y_pred):
    jitter_points = rand_jitter(np.zeros(len(y_pred)))
    hues = y_pred.apply(lambda x: 'green' if x else 'red')

    plt.figure(figsize=(40, 10))
    plt.plot([0, 0], [-0.002, 0.002], '-')
    plt.scatter(y_trans, jitter_points, c=hues, marker='.')
    plt.xlabel('Power transformed days on tx')
    plt.ylabel('(random jitter)')

def regression_plot(y_trans, y_pred):
    plt.figure(figsize=(10, 10))
    plt.plot(y_trans, y_pred, '.')
    point1 = min(np.append(y_trans, y_pred))
    point2 = max(np.append(y_trans, y_pred))
    plt.plot([point1, point2], [point1, point2], '-')
    plt.xlabel('Power transformed days on tx')
    plt.ylabel('Predicted power transformed days on tx')

def main():
    local_time = time.localtime()
    output_dir_name = "{}_{}_{}".format(local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to prediction results tsv to visualize. Must have columns: \'y\', \'y_trans\' \'y_pred\'', required=True)
    parser.add_argument('-o', '--out_dir', help='Directory where to save the output files. Default: {}'.format(output_dir_name), default=output_dir_name)

    args = parser.parse_args()
    input_path = args.input
    out_dir = args.out_dir

    parsed_input = input_path.split('/')
    input_file_name = parsed_input[len(parsed_input)-1]
    input_file_prefix = input_file_name.split('.')[0]
    cancer_type = input_file_prefix.split('_')[0]
    drug_name = input_file_prefix.split('_')[1]

    results_df = pd.read_csv(input_path, index_col=0, sep='\t')
    test_set_mask = np.logical_not(pd.isnull(results_df['y_pred']))
    test_set_df = results_df[test_set_mask]

    if len(np.unique(test_set_df['y_pred'])) > 2:
        discrete = False
    else:
        discrete = True
    
    if discrete:
        discrete_plot(test_set_df['y_trans'], test_set_df['y_pred'])
    else:
        regression_plot(test_set_df['y_trans'], test_set_df['y_pred'])

    plt.title('Cancer type: {} vs Drug: {}'.format(cancer_type, drug_name))
    plt.savefig(out_dir + '/{}.png'.format(input_file_prefix))

if __name__ == '__main__':
    main()