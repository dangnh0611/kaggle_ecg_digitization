import argparse

import timm


def parse_args():
    args = argparse.ArgumentParser("Download online resource.")
    args.add_argument("--repo", type=str, default="timm")
    args.add_argument("--card", type=str)
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.repo == "timm":
        model = timm.create_model(args.card, pretrained=True)
        print(model)
    elif args.repo == "hf":
        raise NotImplementedError
    else:
        raise ValueError