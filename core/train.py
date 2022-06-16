from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F

from core.test import AverageMeter
from metrics.metrics import dice_coeff, get_ece, plot_ece_frequency, reliability_diagram


def get_dice(labels, outputs):
    dice_arr_per_label_type = []
    for j in range(labels.size()[0]):
        label = labels.cpu().data[j].squeeze()
        label = label.squeeze().cpu().numpy()
        pred_ = outputs.cpu().data[j]

        pred = pred_.squeeze()
        out2 = pred_.data.max(0)[1].squeeze_(1)
        # print(outputs.shape, pred_.shape, out2.shape, out2.unique())
        for label_type in range(1, pred.shape[0], 1):
            dice_arr_per_label_type.append(dice_coeff(label == label_type, out2 == label_type))

    return dice_arr_per_label_type


def get_temperature_scaling(teacher_model, dataloaders, args):
    teacher_model.eval()
    loss_ce = torch.nn.CrossEntropyLoss()
    best_ece = np.inf
    opt_T = 1
    const_loss = 0
    start = 2.45
    inc = 0.01
    end = 2.55
    T_list = np.arange(start, end + inc, inc)
    print("T, ECE, Optimal T, Best ECE")
    with torch.no_grad():
        for T in T_list:
            ece, acc_all, conf_all, bm_all = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            losses = AverageMeter()
            dice_arr = AverageMeter()
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.cuda(), labels.cuda()
                teacher_out = teacher_model(inputs)
                dice_arr.update(np.mean(get_dice(labels, teacher_out)), inputs.size(0))
                loss = loss_ce(teacher_out, labels.long())
                losses.update(loss.item(), inputs.size(0))
                pred_conf = F.softmax(teacher_out / T, dim=1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                ece_per_batch, acc, conf, bm = get_ece(pred_conf, labels, ignore_bg=False)
                ece.update(ece_per_batch, inputs.size(0))
                acc_all.update(acc, inputs.size(0))
                conf_all.update(conf, inputs.size(0))
                bm_all.update(bm, inputs.size(0))

            ece_avg = ece.avg
            acc_avg = acc_all.avg
            conf_avg = conf_all.avg
            bm_avg = bm_all.avg
            losses_avg = np.round(losses.avg, 4)
            dice_avg = dice_arr.avg

            if T == start:
                best_ece = ece_avg
                opt_T = T
                const_loss = losses_avg
                reliability_diagram(conf_avg, acc_avg, title=T)
                plot_ece_frequency(bm_avg, title=T)

            if const_loss != losses_avg:
                print("Loss is changing! Something is wrong...")
                break
            if ece_avg < best_ece and const_loss == losses_avg and T is not 1:
                best_ece = ece_avg
                opt_T = T
            reliability_diagram(conf_avg, acc_avg, title=T)
            plot_ece_frequency(bm_avg, title=T)
            # print(
            #     "current loss:%.4f, ece:%.4f, T:%.2f, best ece:%.4f, opt_T:%.2f, const_loss:%.4f, dice:%.4f"
            #     % (losses_avg, ece_avg, T, best_ece, opt_T, const_loss, dice_avg)
            # )
            print(f"{T}, {ece_avg}, {opt_T}, {best_ece}")
            # break

    # print(ece_avg)


def get_threshold(teacher_model, dataloaders, args):
    teacher_model.eval()
    bin_size = 1000
    conf_values = np.arange(1 / bin_size, 1 + 1 / bin_size, 1 / bin_size)
    hist_prob = torch.zeros(len(conf_values))
    mu, mu_update = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.cuda(), labels.cuda()
            teacher_out = teacher_model(inputs)
            out_confidence = F.softmax(teacher_out / args.opt_t, dim=1)
            labels_oh = F.one_hot(labels.to(torch.int64), num_classes=args.num_classes).contiguous().permute(0, 3, 1, 2)
            true_prob_all = out_confidence * labels_oh
            true_prob, _ = true_prob_all.cpu().max(dim=1, keepdim=True)
            hist_prob += torch.histc(true_prob, bins=bin_size, min=0, max=1)

    hist_prob_norm = hist_prob / hist_prob.sum()
    hist_prob_perc = [round(1 - item.item(), 3) for item in accumulate(hist_prob_norm)]
    # print(hist_prob_perc)
    for idx in range(len(hist_prob_perc) - 1, 0, -1):
        if hist_prob_perc[idx] > args.beta:
            mu = conf_values[idx]
            print("percentage of pixels found:", hist_prob_perc[idx])
            break
    mu_update = mu / (args.epochs * args.gamma / args.mu_update_every)
    return mu, mu_update


def get_threshold_cpl(teacher_model, dataloaders, args):
    teacher_model.eval()
    bin_size = 1000
    conf_values = np.arange(1 / bin_size, 1 + 1 / bin_size, 1 / bin_size)
    hist_prob = torch.zeros(len(conf_values))
    mu, mu_update = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.cuda(), labels.cuda()
            teacher_out = teacher_model(inputs)
            out_confidence = F.softmax(teacher_out / args.opt_t, dim=1)
            labels_oh = F.one_hot(labels.to(torch.int64), num_classes=args.num_classes).contiguous().permute(0, 3, 1, 2)
            true_prob_all = out_confidence * labels_oh
            true_prob, _ = true_prob_all.cpu().max(dim=1, keepdim=True)
            hist_prob += torch.histc(true_prob, bins=bin_size, min=0, max=1)

    hist_prob_norm = hist_prob / hist_prob.sum()
    hist_prob_perc = [round(1 - item.item(), 3) for item in accumulate(hist_prob_norm)]
    # print(hist_prob_perc)
    for idx in range(len(hist_prob_perc) - 1, 0, -1):
        if hist_prob_perc[idx] > args.cpl_beta:
            mu = conf_values[idx]
            print("percentage of pixels found CPL:", hist_prob_perc[idx])
            break
    mu_update = mu / (args.epochs * args.cpl_gamma / args.mu_update_every)
    return mu, mu_update


def step_train(model, dataloaders, loss_func, optimizer):
    model.train()
    losses = []
    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels.long())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def step_valid(model, dataloaders):
    model.eval()
    dice_arr = AverageMeter()
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            dice_arr.update(np.mean(get_dice(labels, outputs)), inputs.size(0))
    return dice_arr.avg


def training_iid(model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20, ckpt_dir=None, writer=None):
    best_dice = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        train_loss = step_train(model, dataloaders, loss_func, optimizer)
        dice_arr = step_valid(model, dataloaders)
        epoch_dice = np.mean(dice_arr)
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.state_dict(), ckpt_dir)
            best_epoch = epoch
        logs = "Epoch:{}/{}, Loss_T:{:.4f}, Dice: {:.4f}, Best epoch:{}, Best Dice: {:.4f}".format(
            epoch + 1, num_epochs, train_loss, epoch_dice, best_epoch, best_dice
        )
        print(logs)
        writer.add_scalar("Dice", epoch_dice, epoch)
        writer.flush()
    writer.close()


def step_train_kd(teacher_model, student_model, dataloaders, loss_func, optimizer):
    student_model.train()
    losses = []
    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = student_model(inputs)
        with torch.no_grad():
            teacher_out = teacher_model(inputs)

        loss = loss_func.kd(outputs, teacher_out, labels.long())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def training_kd(
    teacher_model,
    student_model,
    dataloaders,
    loss_func,
    optimizer,
    scheduler,
    num_epochs=20,
    ckpt_dir=None,
    writer=None,
):
    teacher_model.eval()
    best_dice = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        train_loss = step_train_kd(teacher_model, student_model, dataloaders, loss_func, optimizer)
        dice_arr = step_valid(student_model, dataloaders)
        epoch_dice = np.mean(dice_arr)
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(student_model.state_dict(), ckpt_dir)
            best_epoch = epoch

        print(
            "Epoch:{}/{}, Loss_T:{:.4f}, Dice: {:.4f}, Best epoch:{}, Best Dice: {:.4f}".format(
                epoch + 1, num_epochs, train_loss, epoch_dice, best_epoch, best_dice
            )
        )
        writer.add_scalar("Dice", epoch_dice, epoch)
        writer.flush()
    writer.close()


def step_train_cd(
    teacher_model,
    student_model,
    dataloaders,
    loss_func,
    optimizer,
    mu,
    opt_t,
    cd_mode,
    cpl_weights_to,
    alpha,
    cpl_mu,
):
    student_model.train()
    losses = []
    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = student_model(inputs)
        with torch.no_grad():
            teacher_out = teacher_model(inputs)

        loss = loss_func.cd(
            outputs,
            teacher_out,
            labels.long(),
            mu,
            opt_t,
            cd_mode,
            cpl_weights_to,
            alpha,
            cpl_mu,
        )
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def training_cd(
    teacher_model,
    student_model,
    dataloaders,
    loss_func,
    optimizer,
    scheduler,
    args,
    writer=None,
):
    print("{} and mode {} training start!!!".format(args.train_opt, args.cd_mode))
    teacher_model.eval()
    best_dice = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = step_train_cd(
            teacher_model,
            student_model,
            dataloaders,
            loss_func,
            optimizer,
            args.mu,
            args.opt_t,
            args.cd_mode,
            args.cpl_weights_to,
            args.alpha,
            args.cpl_mu,
        )
        dice_arr = step_valid(student_model, dataloaders)
        epoch_dice = np.mean(dice_arr)
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(student_model.state_dict(), args.model_path)
            best_epoch = epoch
        if epoch % args.mu_update_every == 0 and epoch is not 0:  # Update regularly
            args.mu -= args.mu_update
            args.mu = max(args.mu, 0)
            print("epoch:{}, mu:{:.4f}".format(epoch, args.mu))
        if epoch % args.cpl_mu_update_every == 0 and epoch is not 0:  # Update regularly
            args.cpl_mu -= args.cpl_mu_update
            args.cpl_mu = max(args.cpl_mu, 0)
            print("epoch:{}, cpl_mu:{:.4f}".format(epoch, args.cpl_mu))

        print(
            "Epoch:{}/{}, Loss_T:{:.4f}, Dice: {:.4f}, Best epoch:{}, Best Dice: {:.4f}".format(
                epoch + 1, args.epochs, train_loss, epoch_dice, best_epoch, best_dice
            )
        )
        writer.add_scalar("Dice", epoch_dice, epoch)
        writer.add_scalar("Mu", args.mu, epoch)
        writer.add_scalar("CPL Mu", args.mu, epoch)
        writer.flush()
    writer.close()
