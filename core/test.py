import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics.metrics import dice_coeff, iou, precision


def get_dice(labels, outputs):
    dice_arr_per_label_type = {each_label: [] for each_label in range(1, outputs.shape[1], 1)}
    iou_arr_per_label_type = {each_label: [] for each_label in range(1, outputs.shape[1], 1)}
    precision_arr_per_label_type = {each_label: [] for each_label in range(1, outputs.shape[1], 1)}
    for j in range(labels.size()[0]):
        label = labels.cpu().data[j].squeeze()
        label = label.squeeze().cpu().numpy()
        pred_ = outputs.cpu().data[j]
        pred = pred_.squeeze()
        out2 = pred_.data.max(0)[1].squeeze_(1)
        for label_type in range(1, pred.shape[0], 1):
            dice_arr_per_label_type[label_type].append(dice_coeff(label == label_type, out2 == label_type))
            iou_arr_per_label_type[label_type].append(iou(label == label_type, out2 == label_type))
            precision_arr_per_label_type[label_type].append(precision(label == label_type, out2 == label_type))

    return dice_arr_per_label_type, iou_arr_per_label_type, precision_arr_per_label_type


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test(model, dataloaders, num_classes, save_results):
    was_training = model.training
    model.eval()

    n_classes = num_classes  # model.module.out.out_channels

    dice_arr = {each_label: AverageMeter() for each_label in range(1, n_classes, 1)}
    iou_arr = {each_label: AverageMeter() for each_label in range(1, n_classes, 1)}
    precision_arr = {each_label: AverageMeter() for each_label in range(1, n_classes, 1)}

    print(len(dataloaders["val"].dataset))
    if save_results:
        complete_metrics = []
        os.makedirs(save_results, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(dataloaders["val"]):
            inputs = sample[0].cuda()
            labels = sample[1].cuda()

            outputs = model(inputs)

            dice_arr_per_label_type, iou_arr_per_label_type, precision_arr_per_label_type = get_dice(labels, outputs)
            for label_type in range(1, n_classes):
                dice_arr[label_type].update(
                    np.mean(dice_arr_per_label_type[label_type]),
                    len(dice_arr_per_label_type[label_type]),
                )
                iou_arr[label_type].update(
                    np.mean(iou_arr_per_label_type[label_type]),
                    len(iou_arr_per_label_type[label_type]),
                )
                precision_arr[label_type].update(
                    np.mean(precision_arr_per_label_type[label_type]),
                    len(precision_arr_per_label_type[label_type]),
                )

            if save_results:
                img_paths = sample[2]
                for j, (output, each_img_path) in enumerate(zip(F.softmax(outputs, dim=1), img_paths)):
                    dice_single = np.mean([dice_per_label[j] for dice_per_label in dice_arr_per_label_type.values()])
                    iou_single = np.mean([iou_per_label[j] for iou_per_label in iou_arr_per_label_type.values()])
                    precision_single = np.mean(
                        [precision_per_label[j] for precision_per_label in precision_arr_per_label_type.values()]
                    )
                    img = output.argmax(dim=0)

                    def tensor_to_img(tensor):
                        if "seq_" in each_img_path:  # robo dataset
                            labels_json = json.load(open("datasets/robo_labels.json"))
                            class_to_color = {}
                            for each_class in labels_json:
                                class_to_color[each_class["classid"]] = each_class["color"]
                            h, w = tensor.shape
                            image_tensor = torch.zeros(h, w, 3, dtype=torch.long)
                            for each_class in class_to_color:
                                idx = tensor == each_class
                                image_tensor[idx] = torch.tensor(class_to_color[each_class], dtype=torch.long)
                            image_tensor = torch.flip(image_tensor, dims=[2])
                        else:  # bus dataset
                            image_tensor = tensor * 255
                        return image_tensor.detach().cpu().numpy()

                    each_img_path = os.path.join(save_results, each_img_path.replace("/", "_"))
                    cv2.imwrite(each_img_path, tensor_to_img(img))

                    metrics = {
                        "name": each_img_path,
                        "dice": dice_single,
                        "iou": iou_single,
                        "precision": precision_single,
                    }

                    complete_metrics.append(metrics)

        if save_results:
            keys = complete_metrics[0].keys()
            with open(os.path.join(save_results, "results.csv"), "w", newline="") as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(complete_metrics)

        model.train(mode=was_training)

    final_dice = [dice_arr[label_type].avg for label_type in range(1, n_classes)]
    final_dice.append(np.mean(final_dice))
    final_iou = [iou_arr[label_type].avg for label_type in range(1, n_classes)]
    final_iou.append(np.mean(final_iou))
    final_precision = [precision_arr[label_type].avg for label_type in range(1, n_classes)]
    final_precision.append(np.mean(final_precision))
    return {
        "dice": final_dice,
        "iou": final_iou,
        "precision": final_precision,
        # "recall": np.mean(recall_arr),
        # "f1": f1(np.mean(precision_arr), np.mean(recall_arr)),
    }


def visualize_results(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():

        figure, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(15, 3.75 * num_images))

        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                img = inputs.cpu().data[j].squeeze()
                label = labels.cpu().data[j].squeeze()
                label = label.squeeze().cpu().numpy()
                pred = outputs.cpu().data[j].squeeze()
                pred = nn.Softmax()(pred)[1]
                pred = pred.squeeze().cpu().numpy()

                axes[j, 0].imshow(np.transpose(img, (1, 2, 0)), cmap="gray")
                axes[j, 1].imshow(pred, cmap="gray")
                pred[pred > 0.5] = 255
                pred[pred <= 0.5] = 0
                # post_process = max_contour(pred)
                axes[j, 2].imshow(pred, cmap="gray")
                axes[j, 3].imshow(label, cmap="gray")
                cols = ["Input", "Prediction", "Post-Process", "Ground Truth"]

                for ax, col in zip(axes[0], cols):
                    ax.set_title(col)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    figure.tight_layout()
                    return
        model.train(mode=was_training)
