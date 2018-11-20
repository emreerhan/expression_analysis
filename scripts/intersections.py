import pandas as pd
import numpy as np 
import itertools
import argparse

def main():
    test_sets = [{1,2,3,4}, {1,2,3}, {3,4,5,6}, {4,5}, {2,3,4}, {2,1,5}, {1,2,3,4}]
    df = get_intersections(test_sets)
    df.to_csv('test.tsv', sep='\t')

def get_intersection(sets, n_sets):
    intersections_df = pd.DataFrame()
    sets = np.array(sets)
    for set_index_combo in itertools.combinations(range(len(sets)), n_sets):
        combo_cardinality = len(set.intersection(*sets[list(set_index_combo)]))
        row = np.zeros(len(sets)+1)
        row[len(row)-1] = combo_cardinality
        row[list(set_index_combo)] = True
        intersections_df = intersections_df.append(pd.Series(row), ignore_index = True)
    return intersections_df

def get_intersections(sets):
    # columns = ["set{}".format(i) for i in range(len(sets))]
    intersections_df = pd.DataFrame()
    sets = np.array(sets)
    for i in range(1, len(sets)):
        intersection_df = get_intersection(sets, i)
        intersections_df = intersections_df.append(intersection_df, ignore_index=True)

    return intersections_df

if __name__ == '__main__':
    main()