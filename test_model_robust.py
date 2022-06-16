import argparse
import csv
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from core.test import test
from data.dataloader import LoadImageDataset, LoadRoboDataset
from train_model import get_model
from utils.network_utils import remove_dataparallel_wrapper, set_seed

warnings.filterwarnings("ignore")


def write_csv(filename, data):
    with open(filename, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


def get_corrupted_datapath(val_paths, filter, severity, phase="val"):
    for idx, path in enumerate(val_paths):
        val_paths[idx][0] = os.path.join(os.path.dirname(path[0]) + "_c", filter, severity, os.path.basename(path[0]))
    return val_paths


def get_data(args, filter, severity):
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)

    train_paths = json.load(open(os.path.join("datasets", args.data + "_train.json")))
    val_paths = json.load(open(os.path.join("datasets", args.data + "_val.json")))
    if severity != 0:
        val_paths = get_corrupted_datapath(val_paths, filter, str(severity), phase="val")
    apply_transform = True if args.data == "needle" else False
    if args.data == "robo":
        args.num_classes = 12
        labels_json = json.load(open(os.path.join("datasets", args.data + "_labels.json")))
        labels = {}
        for label_type in labels_json:
            labels[label_type["classid"]] = label_type["color"]
        train_data = LoadRoboDataset(
            train_paths, labels, args.data_root, args.img_size, apply_transform=apply_transform, phase="train"
        )
        val_data = LoadRoboDataset(val_paths, labels, args.data_root, args.img_size, apply_transform=False, phase="val")
    else:
        args.num_classes = 2
        train_data = LoadImageDataset(train_paths, args.data_root, args.img_size, apply_transform=apply_transform)
        val_data = LoadImageDataset(val_paths, args.data_root, args.img_size, apply_transform=False)

    dataloaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
    }

    return dataloaders


def arg_parser():
    parser = argparse.ArgumentParser(description="Curriculum KD")
    parser.add_argument("--data", default="bus", help="options:[needle, bus]")
    parser.add_argument("--data-root", default=None, help="path that has the dataset images")
    parser.add_argument("--img-size", default=224, type=int, help="img size")
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--model-name", default="UNet", help="options: [LinkNet, UNet, CKD, KD, DeepLab]")
    parser.add_argument("--model-path", default="outputs/bus/unet/unet.pth", help="Path to trained model weights")
    parser.add_argument("--num-classes", default=2, type=int, help="number of classes")
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)

    results_file = "robo_valid_robust.csv"
    filters = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    severities = [0, 1, 2, 3, 4, 5]
    #
    args.model_path = [
        "outputs/robo/iid-mob/unet.pth",
        "outputs/robo/kd-mob/kd_best.pth",
        "outputs/robo/cd-tpl-mob/tpl_unet_085_06.pth",
    ]
    model_list = ["unet", "kd", "tpl"]
    for m_idx, model_name in enumerate(model_list):
        print("model path found:", args.model_path[m_idx])
        write_csv(results_file, [model_name, "0", "1", "2", "3", "4", "5", "Avg"])
        seg_model = get_model(args)
        seg_model = seg_model.to(device)
        try:
            seg_model.load_state_dict(torch.load(args.model_path[m_idx], map_location=device))
        except RuntimeError:
            seg_model.load_state_dict(
                remove_dataparallel_wrapper(torch.load(args.model_path[m_idx], map_location=device))
            )
        if device == torch.device("cuda"):
            seg_model = nn.DataParallel(seg_model).to(device)

        for filter in filters:
            dice_all = []
            for severity in severities:
                dataloaders = get_data(args, filter, severity)
                metrics = test(seg_model, dataloaders, args.num_classes)
                val_dice = np.mean(metrics["dice"])
                dice_all.append("{:.4f}".format(val_dice))
                print("filter:{}, severity:{}, valid_dice:{:.4f}".format(filter, severity, val_dice))
            write_csv(
                results_file, [filter, dice_all[0], dice_all[1], dice_all[2], dice_all[3], dice_all[4], dice_all[5]]
            )


if __name__ == "__main__":
    set_seed(12345)
    parsed = arg_parser().parse_args()
    main(parsed)
