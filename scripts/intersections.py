import pandas as pd
import numpy as np 
import itertools

def main():
    test_sets = [{1,2,3,4}, {1,2,3}, {3,4,5,6}]
    df = get_intersections(test_sets)
    df.write_csv('test.tsv', sep='\t')

def get_intersections(sets):
    columns = ["set{}".format(i) for i in range(len(sets))]
    intersections_df = pd.DataFrame(columns = columns)

    for i in len(sets):
        for set_index_combo in itertools.combinations(len(sets), i):
            combo_size = len(set.intersection(sets[set_index_combo]))
            row = np.ones(len(sets), dtype=np.bool)
            row[set_index_combo] = False
            intersections_df.append(row)
    return intersections_df

if __name__ == '__main__':
    main()