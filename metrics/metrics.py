import os

import matplotlib.pyplot as plt
import numpy as np


def plot_ece_frequency(Bm_avg, title=None, n_bins=10):
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    indx = np.arange(0, 1.1, 1 / n_bins)
    plt.xticks(indx)
    plt.title(f"{title:3f}")
    plt.bar(indx[:-1], Bm_avg / Bm_avg.sum(), width=0.08, align="edge")
    if not os.path.exists("ece"):
        os.makedirs("ece")
    plt.savefig("ece/ece_frequency_{:.3f}.png".format(title), dpi=300)
    plt.clf()


def reliability_diagram(conf_avg, acc_avg, title=None, leg_idx=0, n_bins=10):
    plt.figure(2)
    # plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot([conf_avg[acc_avg > 0][0], 1], [conf_avg[acc_avg > 0][0], 1], linestyle="--")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    # plt.xticks(np.arange(0, 1.1, 1/n_bins))
    # plt.title(title)
    plt.plot(conf_avg[acc_avg > 0], acc_avg[acc_avg > 0], marker=".", label=title)
    plt.legend()
    if not os.path.exists("ece"):
        os.makedirs("ece")
    plt.savefig("ece/ece_reliability_{:.3f}.png".format(title), dpi=300)


def get_ece(preds, targets, n_bins=10, ignore_bg=False):
    # ignore_bg = False to ignore bckground class from ece calculation
    bg_cls = 0 if ignore_bg else -1
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    confidences, predictions = confidences[targets > bg_cls], predictions[targets > bg_cls]
    accuracies = predictions == targets[targets > bg_cls]
    # print('accuracies',accuracies.shape, accuracies.mean() )
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        Bm[bin_idx] = bin_size
        if bin_size > 0:
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        bin_idx += 1

    ece_all = Bm * np.abs((acc - conf)) / Bm.sum()
    ece = ece_all.sum()
    return ece, acc, conf, Bm


def dice_coeff(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    smooth = 0.0001
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    if np.isnan(iou_score):
        iou_score = 1
    return iou_score


def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred))
    fn = np.sum(np.logical_and(y_true, 1 - y_pred))
    recall_val = tp / (tp + fn)
    if np.isnan(recall_val):
        recall_val = 1
    return recall_val


def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(1 - y_true, y_pred))
    precision_val = tp / (tp + fp)
    if np.isnan(precision_val):
        precision_val = 1
    return precision_val


def f1(precision_val, recall_val):
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    return f1_val


def angle_acc(act_angle, pred_angle):
    return (act_angle - pred_angle) ** 2


def dist_acc(pred_point1, pred_point2, act_point1, act_point2, size):

    center = np.array(size) / 2

    pred_point1 = np.array(pred_point1)
    pred_point2 = np.array(pred_point2)

    pred_dist = np.cross(pred_point2 - pred_point1, center - pred_point1) / np.linalg.norm(pred_point2 - pred_point1)

    act_point1 = np.array(act_point1)
    act_point2 = np.array(act_point2)

    act_dist = np.cross(act_point2 - act_point1, center - act_point1) / np.linalg.norm(act_point2 - act_point1)

    return np.square(act_dist - pred_dist)
