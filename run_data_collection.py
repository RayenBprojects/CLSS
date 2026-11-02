from core import collect_kitti_data

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clean_existing_files", action= 'store_true')
    args= parser.parse_args()

    collect_kitti_data(args.clean_existing_files)



if __name__ == "__main__":
    main()