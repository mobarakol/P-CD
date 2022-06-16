import argparse
import json
import os
import warnings

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.train import get_threshold, get_threshold_cpl, training_cd, training_iid, training_kd
from data.dataloader import LoadImageDataset, LoadRoboDataset
from metrics.losses import loss_functions
from models.deeplab import DeepLabV3
from models.linknet import LinkNet
from models.unet import UNet
from utils.network_utils import remove_dataparallel_wrapper, set_seed

warnings.filterwarnings("ignore")


def arg_parser():
    parser = argparse.ArgumentParser(description="Curriculum KD")

    # Data
    parser.add_argument("--data", default="bus", help="options:[needle, bus, robo]")
    parser.add_argument("--data-root", default=None, help="path that has the dataset images")
    parser.add_argument("--img-size", default=224, type=int, help="img size")
    parser.add_argument("--num-classes", default=2, type=int, help="number of classes")
    parser.add_argument("--batch-size", default=24, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.003, type=float, help="initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--model-path", default="outputs/bus/ckd/unet.pth", help="path where model is to be saved")
    parser.add_argument(
        "--teacher-path", default="outputs/bus/iid-sps-v7/unet.pth", help="path to pretrained teacher model"
    )
    parser.add_argument("--tb", action="store_true", help="enable tensorboard logging")

    # General Model
    parser.add_argument("--model-name", default="UNet", help="options:[UNet, LinkNet, DeepLab]")
    parser.add_argument("--train-opt", default="CD", help="options:[IID, KD, CD]")

    # CD
    parser.add_argument("--cd-mode", default="tpl", help="options:[tpl, uncert, cpl, cpl_tpl]")
    parser.add_argument("--mu-update", default=0.1, type=float, help="increment mu every epoch-width")
    parser.add_argument("--mu-update-every", default=5, type=int, help="epoch-width")
    parser.add_argument("--mu", default=1, type=float, help="Initial threshold value")

    parser.add_argument("--alpha", default=0.3, type=float, help="Relative scaling of CE and KLD in CPL")
    parser.add_argument("--beta", default=0.5, type=float, help="Initial percentage of data")
    parser.add_argument("--gamma", default=0.5, type=float, help="Percentage of epoch all data will be used")

    # CPL Independent thresholding
    parser.add_argument("--cpl-beta", default=0.5, type=float, help="Initial percentage of data")
    parser.add_argument("--cpl-gamma", default=0.5, type=float, help="Percentage of epoch all data will be used")
    parser.add_argument("--cpl-mu-update", default=0.1, type=float, help="increment mu every epoch-width")
    parser.add_argument("--cpl-mu-update-every", default=5, type=int, help="cpl mu epoch width")
    parser.add_argument("--cpl-mu", default=1, type=float, help="Initial threshold value")

    # Loss
    parser.add_argument("--cpl-weights-to", default="ce", help="options:[ce, kld, both]")
    parser.add_argument("--kld-loss-temp", default=4.5, type=float, help="temperature")

    # TS
    parser.add_argument("--opt-t", default=2.51, type=float, help="teacher's logits scale to get conf")

    return parser


def get_model(args):
    if args.model_name == "UNet":
        model = UNet(args.num_classes)
    elif args.model_name == "DeepLab":
        model = DeepLabV3(args.num_classes)
    elif args.model_name == "LinkNet":
        model = LinkNet(args.num_classes)
    else:
        raise Exception

    return model


def get_data(args):
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)

    train_paths = json.load(open(os.path.join("datasets", args.data + "_train.json")))
    val_paths = json.load(open(os.path.join("datasets", args.data + "_val.json")))
    apply_transform = True if args.data == "needle" else False
    if args.data == "robo":
        args.num_classes = 12
        labels_json = json.load(open(os.path.join("datasets", args.data + "_labels.json")))
        labels = {}
        for label_type in labels_json:
            labels[label_type["classid"]] = label_type["color"]
        train_data = LoadRoboDataset(
            train_paths,
            labels,
            args.data_root,
            args.img_size,
            apply_transform=apply_transform,
            phase="train",
            return_filenames=getattr(args, "save_results", False),
        )
        val_data = LoadRoboDataset(
            val_paths,
            labels,
            args.data_root,
            args.img_size,
            apply_transform=False,
            phase="val",
            return_filenames=getattr(args, "save_results", False),
        )
    else:
        args.num_classes = 2
        train_data = LoadImageDataset(
            train_paths,
            args.data_root,
            args.img_size,
            apply_transform=apply_transform,
            return_filenames=getattr(args, "save_results", False),
        )
        val_data = LoadImageDataset(
            val_paths,
            args.data_root,
            args.img_size,
            apply_transform=False,
            return_filenames=getattr(args, "save_results", False),
        )

    dataloaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
    }

    return dataloaders


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    dataloaders = get_data(args)
    student_model = get_model(args)
    if device == torch.device("cuda"):
        student_model = nn.DataParallel(student_model).to(device)
    optimizer = Adam(student_model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    writer = SummaryWriter(os.path.dirname(args.model_path))
    if args.train_opt == "IID":
        loss_func = nn.CrossEntropyLoss()
        training_iid(
            student_model,
            dataloaders,
            loss_func,
            optimizer,
            exp_lr_scheduler,
            num_epochs=args.epochs,
            ckpt_dir=args.model_path,
            writer=writer,
        )

    elif args.train_opt in ["CD", "KD"]:
        loss_func = loss_functions(temp=args.kld_loss_temp, num_classes=args.num_classes)
        teacher_model = get_model(args)
        try:
            teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
        except RuntimeError:
            teacher_model.load_state_dict(
                remove_dataparallel_wrapper(torch.load(args.teacher_path, map_location=device))
            )

        teacher_model = nn.DataParallel(teacher_model).to(device)
        if args.train_opt == "KD":
            # from core.train import get_temperature_scaling
            # get_temperature_scaling(teacher_model, dataloaders, args)
            # exit(0)
            training_kd(
                teacher_model,
                student_model,
                dataloaders,
                loss_func,
                optimizer,
                exp_lr_scheduler,
                num_epochs=args.epochs,
                ckpt_dir=args.model_path,
                writer=writer,
            )
        elif args.train_opt == "CD":
            # If TPL or TPL + uncert    : Get `mu` & `mu_update` by histogram ranking and threshold method
            # If CPL                    : Use `mu` & `mu_update` as given in arg parse
            # If CPL+TPL                : Get *shared* `mu` & `mu_update` by histogram ranking and threshold method

            # get_temperature_scaling(teacher_model, dataloaders, args)
            if args.cd_mode == "tpl" or args.cd_mode == "tpl_uncert":
                args.mu, args.mu_update = get_threshold(teacher_model, dataloaders, args)
                print("initial_mu:{:.4f}, mu_update:{:.4f}".format(args.mu, args.mu_update))

            if args.cd_mode == "cpl_tpl":
                args.mu, args.mu_update = get_threshold(teacher_model, dataloaders, args)
                args.cpl_mu, args.cpl_mu_update = get_threshold_cpl(teacher_model, dataloaders, args)
                print("initial_cpl_mu:{:.4f}, cpl_mu_update:{:.4f}".format(args.cpl_mu, args.cpl_mu_update))
                print("initial_mu:{:.4f}, mu_update:{:.4f}".format(args.mu, args.mu_update))

            training_cd(
                teacher_model,
                student_model,
                dataloaders,
                loss_func,
                optimizer,
                exp_lr_scheduler,
                args,
                writer=writer,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # if args.tb:
    #     writer = SummaryWriter(os.path.join(os.path.dirname(args.model_path), "tb_log"))
    # else:
    #     writer = None

    if args.tb:
        writer.close()


if __name__ == "__main__":
    set_seed(12345)
    parsed = arg_parser().parse_args()
    main(parsed)
