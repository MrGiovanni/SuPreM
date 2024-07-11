import os
import random
import argparse

def create_five_fold_lists(args):
    with open(args.input_file1, 'r') as f:
        cases1 = f.read().splitlines()

    with open(args.input_file2, 'r') as f:
        cases2 = f.read().splitlines()

    combined_cases = cases1 + cases2

    random.shuffle(combined_cases)

    num_cases = len(combined_cases)
    fold_size = num_cases // 5
    folds = [combined_cases[i * fold_size:(i + 1) * fold_size] for i in range(5)]

    remaining_cases = combined_cases[5 * fold_size:]
    for i in range(len(remaining_cases)):
        folds[i].append(remaining_cases[i])

    for i in range(5):
        train_cases = [case for j in range(5) if j != i for case in folds[j]]
        test_cases = folds[i]

        train_output_file = os.path.join(args.output_dir, f'jhh_train_fold_{i}.txt')
        test_output_file = os.path.join(args.output_dir, f'jhh_test_fold_{i}.txt')

        with open(train_output_file, 'w') as f:
            for case in train_cases:
                f.write(case + '\n')

        with open(test_output_file, 'w') as f:
            for case in test_cases:
                f.write(case + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file1', required=True, help='Path to the original train list file')
    parser.add_argument('--input_file2', required=True, help='Path to the original test list file')
    parser.add_argument('--output_dir', required=True, help='Directory to save the files for cross validation')

    args = parser.parse_args()
    
    create_five_fold_lists(args)

if __name__ == "__main__":
    main()

