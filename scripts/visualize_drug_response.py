import pandas as pd
import numpy as np
import time
import argparse

from matplotlib import pyplot as plt

def main():
    local_time = time.localtime()
    output_dir_name = "{}_{}_{}".format(local_time.tm_hour, local_time.tm_min, local_time.tm_sec)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to prediction results tsv to visualize. Must have columns: \'y\', \'y_trans\' \'y_pred\'', required=True)
    parser.add_argument('-o', '--out_dir', help='Directory where to save the output files', default=output_dir_name)

    args = parser.parse_args()
    input_path = args.input

    results_df = pd.read_csv(input_path, index_col=0, sep='\t')
    test_set_mask = np.logical_not(pd.isnull(results_df['y_pred']))
    test_set_df = results_df[test_set_mask]

    y = np.zeros(len(test_set_df))
    hues = test_set_df['y_pred'].apply(lambda x: 'green' if x else 'red')

    plt.figure(figsize=(40, 10))
    plt.scatter(test_set_df['y_trans'], x, c=hues)
    plt.savefig()

if __name__ == '__main__':
    main()