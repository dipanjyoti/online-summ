import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np

# import torch

from data import build_data



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # dataset parameters
    parser.add_argument('--dataset_path', default='/home/paul.1164/Paul/summarization/dataset', type=str)
    parser.add_argument('--dataset_file', default='hagupit', choices=['hagupit', 'hyd', 'sandy', 'uk']) 
    

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')

    return parser


def main(args):

    # print(args)
    # device = torch.device(args.device)

    dataset= build_data(args=args)

    print (dataset)
    exit()








if __name__ == '__main__':
    parser = argparse.ArgumentParser('Online summarization script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)