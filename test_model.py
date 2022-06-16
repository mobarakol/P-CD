import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn

from core.test import test
from train_model import get_data, get_model
from utils.network_utils import remove_dataparallel_wrapper, set_seed

warnings.filterwarnings("ignore")


def arg_parser():
    parser = argparse.ArgumentParser(description="Curriculum KD")
    parser.add_argument("--data", default="bus", help="options:[needle, bus]")
    parser.add_argument("--data-root", default=None, help="path that has the dataset images")
    parser.add_argument("--img-size", default=224, type=int, help="img size")
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--model-name", default="UNet", help="options: [LinkNet, UNet, CKD, KD, DeepLab]")
    parser.add_argument("--model-path", default="outputs/bus/unet/unet.pth", help="Path to trained model weights")
    parser.add_argument("--num-classes", default=2, type=int, help="number of classes")
    parser.add_argument("--save-results", default=False, help="save qualitative results and metrics")
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)

    dataloaders = get_data(args)
    seg_model = get_model(args)

    seg_model = seg_model.to(device)
    try:
        seg_model.load_state_dict(torch.load(args.model_path, map_location=device))
    except RuntimeError:
        seg_model.load_state_dict(remove_dataparallel_wrapper(torch.load(args.model_path, map_location=device)))
    if device == torch.device("cuda"):
        seg_model = nn.DataParallel(seg_model).to(device)

    metrics = test(seg_model, dataloaders, args.num_classes, args.save_results)
    print(",".join(metrics.keys()))
    print(",".join([str(np.round(data, 4)) for data in metrics.values()]))
    # print(",".join([str((data)) for data in metrics.values()]))


if __name__ == "__main__":
    set_seed(12345)
    parsed = arg_parser().parse_args()
    main(parsed)
