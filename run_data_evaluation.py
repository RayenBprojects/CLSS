
from pathlib import Path

import dotenv

import os

import torch

from torch.utils.data import DataLoader

from torch import accelerator

import argparse

from core import evaluate_data

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", default= 16)
    parser.add_argument("-p", "--print_batch_data", action= "store_true")
    args= parser.parse_args()

    evaluate_data(projection_type= 'pinhole', batch_size= int(args.batch_size), print_batch_data= args.print_batch_data)
    evaluate_data(projection_type= 'cylindrical', batch_size= int(args.batch_size), print_batch_data= args.print_batch_data)


if __name__ == "__main__":
    main()