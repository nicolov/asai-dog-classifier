#!/usr/bin/env python

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="/Users/niko/Downloads/dogs-vs-cats/train")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "train/cats"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "train/dogs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "validation/cats"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "validation/dogs"), exist_ok=True)

    for i in range(0, 1000):
        subprocess.check_call(
            [
                "cp",
                os.path.join(args.in_dir, f"cat.{i}.jpg"),
                os.path.join(args.out_dir, "train/cats"),
            ]
        )

    for i in range(1000, 1400):
        subprocess.check_call(
            [
                "cp",
                os.path.join(args.in_dir, f"cat.{i}.jpg"),
                os.path.join(args.out_dir, "validation/cats"),
            ]
        )

    for i in range(10000, 11000):
        subprocess.check_call(
            [
                "cp",
                os.path.join(args.in_dir, f"dog.{i}.jpg"),
                os.path.join(args.out_dir, "train/dogs"),
            ]
        )

    for i in range(11000, 11400):
        subprocess.check_call(
            [
                "cp",
                os.path.join(args.in_dir, f"dog.{i}.jpg"),
                os.path.join(args.out_dir, "validation/dogs"),
            ]
        )


if __name__ == "__main__":
    main()
