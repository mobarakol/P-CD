import math

import torch
import torch.nn.functional as F
from torch import nn


def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma ** 2.0
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance + 1e-16)
    )
    return gaussian_kernel / torch.sum(gaussian_kernel)


class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1, 1]) + 1e-16
        gkernel[int(ksize / 2), int(ksize / 2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize / 2)
        self.svls_layer = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=ksize,
            groups=channels,
            bias=False,
            padding=padding,
            padding_mode="replicate",
        )
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False

    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()


class get_svls_label(torch.nn.Module):
    def __init__(self, num_classes=None, sigma=1, ksize=3):
        super(get_svls_label, self).__init__()
        self.cls = num_classes
        self.svls_layer = get_svls_filter_2d(ksize=ksize, sigma=sigma, channels=self.cls)

    def forward(self, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes=self.cls).contiguous().permute(0, 3, 1, 2).float()
        svls_labels = self.svls_layer(oh_labels)
        return svls_labels


class loss_functions:
    def __init__(self, temp=4.5, num_classes=2):
        self.temp = temp  # KLD Loss Temp (Best at 4.5)
        self.classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.kl = nn.KLDivLoss(reduction="none")
        self.label_to_svlslabel = get_svls_label(num_classes=self.classes).cuda()

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels)

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = F.softmax(teacher_out / self.temp, dim=1)
        outputs_temp = F.log_softmax(outputs / self.temp, dim=1)
        kl = self.kl(outputs_temp, teacher_out_temp) * self.temp * self.temp
        kl = torch.mean(kl, 1)
        return kl

    def kd(self, outputs, teacher_out, labels):
        loss = self.kld_loss(teacher_out, outputs).mean() + self.ce_loss(outputs, labels).mean()
        return loss

    def get_weights_tpl_per_px(self, teacher_out, labels, mu, opt_t):  # SPL
        out_confidence = F.softmax(teacher_out / opt_t, dim=1)
        labels_oh = F.one_hot(labels.to(torch.int64), num_classes=self.classes).contiguous().permute(0, 3, 1, 2)
        true_prob_all = out_confidence * labels_oh
        true_prob, _ = true_prob_all.max(dim=1, keepdim=True)
        weights = torch.where(true_prob >= mu, 1.0, 0.0)
        # weights = torch.where(1 - true_prob <= mu, 1.0, 0.0)
        # print(labels_oh.shape, weights.sum().item(), labels_oh.sum().item(), mu, weights.sum().item()/labels_oh.sum().item())
        return weights

    def get_weights_cpl(self, teacher_out, labels, opt_t, mu):
        out_confidence = F.softmax(teacher_out / opt_t, dim=1)
        labels_oh = F.one_hot(labels.to(torch.int64), num_classes=self.classes).contiguous().permute(0, 3, 1, 2)
        true_prob_all = out_confidence * labels_oh
        true_prob, _ = true_prob_all.max(dim=1, keepdim=True)

        # High confidence ones weighted higher while training
        # Mu keeps decreasing
        # confidence >= mu to be *masked in*, else keep weights as mu
        weights = torch.where(true_prob >= mu, 1.0, max(mu - 0.1, 0.0))
        return weights

    def get_weights_label_uncert_per_px(self, labels, mu):
        svls_labels = self.label_to_svlslabel(labels)
        svls_labels_prob = svls_labels.max(1)[0]
        weights = torch.where(svls_labels_prob >= mu, 1.0, 0.0)
        return weights

    def cd(self, outputs, teacher_out, labels, mu, opt_t, cd_mode, cpl_weights_to, alpha, cpl_mu):
        if cd_mode == "tpl":
            tpl_weights = self.get_weights_tpl_per_px(teacher_out, labels, mu, opt_t)
            loss = tpl_weights * (self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels))

        elif cd_mode == "tpl_uncert":
            tpl_weights = self.get_weights_tpl_per_px(teacher_out, labels, mu, opt_t)
            uncert_weight = self.get_weights_label_uncert_per_px(labels, mu)
            loss = (tpl_weights * uncert_weight) * (self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels))

        elif cd_mode == "cpl":
            cpl_weights = self.get_weights_cpl(teacher_out, labels, opt_t, cpl_mu).squeeze()
            if cpl_weights_to == "ce":
                loss = self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels) * cpl_weights
            elif cpl_weights_to == "kld":
                loss = self.kld_loss(teacher_out, outputs) * cpl_weights + self.ce_loss(outputs, labels)
            elif cpl_weights_to == "both":
                loss = (self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels)) * cpl_weights
            else:
                raise NotImplementedError

        elif cd_mode == "cpl_tpl":
            tpl_weights = self.get_weights_tpl_per_px(teacher_out, labels, mu, opt_t)
            cpl_weights = self.get_weights_cpl(teacher_out, labels, opt_t, cpl_mu).squeeze()
            if cpl_weights_to == "ce":
                loss = tpl_weights * (self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels) * cpl_weights)
            elif cpl_weights_to == "kld":
                loss = tpl_weights * (self.kld_loss(teacher_out, outputs) * cpl_weights + self.ce_loss(outputs, labels))
            elif cpl_weights_to == "both":
                loss = tpl_weights * cpl_weights * (self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels))
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        return loss.mean()
