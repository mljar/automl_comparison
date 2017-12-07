__author__ = 'Piotr Plonski'

import argparse
from compute import compute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--package',  help='Run on selected package [auto-sklearn, h2o, mljar]',
                            required=True, choices=['auto-sklearn', 'h2o', 'mljar'])
    parser.add_argument('-d','--dataset',  help='Dataset id', required=True)
    parser.add_argument('-s','--seed',     help='Seed for computation', required=True, type=int)

    args = parser.parse_args()
    compute(args.package, args.dataset, args.seed)

if __name__ == '__main__':
    main()
