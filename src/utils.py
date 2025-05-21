########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import sys

import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.preprocessing import UnNormalize, ToNumpy

debug = False
def logging(mess):
    if debug: print(mess)

def var_to_device(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    if torch.mps.is_available():
        return variable.to("mps")
    else:
        return variable

class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, emb_k, emb_q, labels, epoch, tau=0.1):
        """
        emb_k: the feature bank with the aggregated embeddings over the iterations
        emb_q: the embeddings for the current iteration
        labels: the correspondent class labels for each sample in emb_q
        """
        total_loss = var_to_device(torch.tensor(0.0)) 
        assert (
            emb_q.shape[0] == labels.shape[0]
        ), "mismatch on emb_q and labels shapes!"
        emb_k = F.normalize(emb_k, dim=-1)
        emb_q = F.normalize(emb_q, dim=1)

        for i, emb in enumerate(emb_q):
            label = labels[i]
            if not (255 in label.unique() and len(label.unique()) == 1):
                label[label == 255] = self.n_classes
                label_sq = torch.unique(label, return_inverse=True)[1]
                oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                count = oh_label.view(-1, oh_label.shape[-1]).sum(
                    dim=0
                )  # num of pixels per cl
                pred = emb.permute(1, 2, 0).unsqueeze(-1)
                oh_pred = (
                    pred * oh_label
                )  # (H, W, Nc, Ncp) Ncp num classes present in the label
                
                oh_pred_flatten = oh_pred.view(
                    oh_pred.shape[0] * oh_pred.shape[1],
                    oh_pred.shape[2],
                    oh_pred.shape[3],
                )
                res_raw = oh_pred_flatten.sum(dim=0) / count  # avg feat per class
                res_new = (res_raw[~res_raw.isnan()]).view(
                    -1, self.n_classes
                )  
                # filter out nans given by intermediate classes (present because of oh)
                label_list = label.unique()
                if self.n_classes in label_list:
                    label_list = label_list[:-1]
                    res_new = res_new[:-1, :]

                # temperature-scaled cosine similarity
                final = (var_to_device(res_new) @ var_to_device(emb_k.T)) / tau

                loss = F.cross_entropy(final, label_list)
                total_loss += loss

        return total_loss / emb_q.shape[0]


class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1, n_accumulations=0, mav_squared=True):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.delta = delta
        self.n_accumulations = n_accumulations
        self.acc = n_accumulations
        self.mav_squared = mav_squared
        print(f"setting mav_squared.{mav_squared}")
        self.count = var_to_device(torch.zeros(self.n_classes))  # count for class
        self.features = {
            i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
        }
        self.features2 = {
            i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
        }

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_features2 = None
        self.previous_count = None


    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        # pixel wise prediction w/ softmax
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1) 
        # list of labels present in this ground-truth target tensor
        gt_labels = torch.unique(sem_gt).tolist()
        # let's order by pixels
        logits_permuted = logits.permute(0, 2, 3, 1)
        # for all label classes in this gt target tensor
        for label in gt_labels:
            # if anomaly/void/unlabeled - skip
            if label == 255:
                continue
            # label mask on gt target
            sem_gt_current = sem_gt == label
            # label mask ok prediction (softmax tensor)
            sem_pred_current = sem_pred == label
            # true-positive mask btw this label and predictions
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            # skip if no true-positive available
            if tps_current.sum() == 0:
                continue
            # get logtits where true-positives
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # get mean of true pos logits
            avg_mav = torch.mean(logits_tps, dim=0)
            # get number of true positives
            n_tps = logits_tps.shape[0]
            ####
            # welford alg https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            ###
            # cumulate number of tp found
            alpha = 0.1
            self.count[label] += 1
            #-> let's calc features contributions as well
            features_contrib = self.features[label]
            # get new data delta from mean
            delta = (logits_tps.mean(dim=0) - features_contrib)
            # get new mean for label - add new contributions
            self.features[label] = (self.features[label] + delta  / (self.count[label] + 1e-8))  if (self.features[label] != var_to_device(torch.zeros(self.n_classes))).any() else logits_tps.mean(dim=0) # instead of: self.features[label] + (delta \ count)
            #-> let's calc features contributions as well
            features_contrib = self.features[label] 
            # get new delta from new mean
            delta2 = (logits_tps.mean(dim=0) - features_contrib)
            # accumulate delta differences
            self.features2[label] = ((self.features2[label] + delta * delta2)) if (self.features2[label] != var_to_device(torch.zeros(self.n_classes))).any() else (delta * delta2)
            
    def forward(
        self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: torch.bool  
    ) -> torch.Tensor:
        if is_train:
            sem_gt = sem_gt.type(torch.uint8)
            # update mav only at training time
            self.cumulate(logits, sem_gt)
            # do it only for 
            self.acc -= 1
                
        if self.acc <= 0:
            self.previous_features = self.features
            self.previous_features2 = self.features2
            self.previous_count = self.count
            # restart accumulating
            self.acc = self.n_accumulations
            # reset  
            self.count = var_to_device(torch.zeros(self.n_classes))  # count for class
            self.features = {i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)}
            self.features2 = {i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)}

        
        if self.previous_features is None:
            return var_to_device(torch.tensor(0.0))

        # list of labels present in this ground-truth target tensor
        gt_labels = torch.unique(sem_gt).tolist()
        # let's order by pixels
        logits_permuted = logits.permute(0, 2, 3, 1)

        acc_loss = var_to_device(torch.tensor(0.0))
        
        for label in gt_labels[:-1]:
            # finalize accumulations
            mav = self.previous_features[label]
            var =  self.previous_features2[label] / (self.previous_count[label] + 1e-8)
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                num = (self.criterion(logs, mav) ** 2 ) if self.mav_squared else (self.criterion(logs, mav))
                den = (var[label] ** 2 + 1e-8)     if self.mav_squared else (var[label] + 1e-8)
                ew_l1 = num / den # squared
                ew_l1_mean = ew_l1.mean()
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                acc_loss += ew_l1_mean # instead of mean
        return acc_loss

    def update(self):
        zeros = {i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)}
        if self.previous_features is not None:
            return torch.stack(list(self.previous_features.values())), torch.stack(list(self.previous_features2.values())) / (self.previous_count + 1e-8)
        return zeros, zeros
    
    def read(self):
        zeros = var_to_device(torch.zeros(self.n_classes, self.n_classes))
        # return mav tensors
        if self.previous_features is not None:
            return torch.stack(list(self.previous_features.values())) , torch.stack(list(self.previous_features2.values())) / (self.previous_count + 1e-8)
        return zeros, zeros



class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, sem_gt):
        # logging("-> ObjectosphereLoss forward")
        logits_unk = logits.permute(0, 2, 3, 1)[torch.where(sem_gt == 255)]
        logits_kn = logits.permute(0, 2, 3, 1)[torch.where(sem_gt != 255)]

        if len(logits_unk):
            loss_unk = (torch.linalg.norm(logits_unk, dim=1)**2).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - (torch.linalg.norm(logits_kn, dim=1)**2)).mean()
        else:
            loss_kn = torch.tensor(0)

        loss = 10 * loss_unk + loss_kn
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device) if device != torch.device("mps") else torch.tensor(weight).to(dtype=torch.float32).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="none",
            ignore_index=-1,
        )
        self.ce_loss.to(device) if device != torch.device("mps") else self.ce_loss.to(dtype=torch.float32).to(device)

    def forward(self, inputs, targets):
        # logging("-> CrossEntropyLoss2d forward")
        losses = []
        targets_m = targets.clone()
        if targets_m.sum() == 0:
            import ipdb;ipdb.set_trace()  # fmt: skip
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(
            targets.flatten().type(self.dtype), minlength=self.num_classes
        )
        # import pandas as pd

        divisor_weighted_pixel_sum = torch.sum(
            number_of_pixels_per_class[1:] * self.weight
        )  # without void
        if divisor_weighted_pixel_sum > 0:
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
        else:
            losses.append(var_to_device(torch.tensor(0.0)))

        return losses


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(), reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device) if device != torch.device("mps") else self.ce_loss.to(dtype=torch.float32).to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None, reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device) if device != torch.device("mps") else self.ce_loss.to(dtype=torch.float32).to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets_m >= 0)  # only non void pixels

    def compute_whole_loss(self):
        return (
            self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy().item()
        )

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0


def print_log(
    epoch, local_count, count_inter, dataset_size, loss, time_inter, learning_rates
):
    print_string = "Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]".format(
        epoch, local_count, dataset_size, 100.0 * local_count / dataset_size
    )
    for i, lr in enumerate(learning_rates):
        print_string += "   lr_{}: {:>6}".format(i, round(lr, 10))
    print_string += "   Loss: {:0.6f}".format(loss.item())
    print_string += "  [{:0.2f}s every {:>4} data]".format(time_inter, count_inter)
    print(print_string, flush=True)


def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print("{:>2} has been successfully saved".format(path))


def save_ckpt_every_epoch(
    ckpt_dir, model, optimizer, epoch, best_miou, best_miou_epoch, mavs, stds
):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_miou_epoch": best_miou_epoch,
        "mavs": mavs,
        "stds": stds,
    }
    ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == "cuda":
            checkpoint = torch.load(model_file, map_location=device)
        else:
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage
            )

        mav_dict = checkpoint["mavs"]
        std_dict = checkpoint["stds"]

        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                model_file, checkpoint["epoch"]
            )
        )
        epoch = checkpoint["epoch"]
        if "best_miou" in checkpoint:
            best_miou = checkpoint["best_miou"]
            print("Best mIoU:", best_miou)
        else:
            best_miou = 0

        if "best_miou_epoch" in checkpoint:
            best_miou_epoch = checkpoint["best_miou_epoch"]
            print("Best mIoU epoch:", best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch, mav_dict, std_dict
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key="mIoU_test"):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, "logs.csv")
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
    assert ckpt_path is not None, f"No trainings found at {ckpt_dir}"
    assert os.path.exists(ckpt_path), f"There is no weights file named {ckpt_path}"
    print(f"Best mIoU: {100*miou:0.2f} at epoch: {epoch}")
    return ckpt_path


def justify(text, length):
    if len(text) >= length:
        return text
    text = text.ljust(len(text)+(length-len(text))//2) # add half spaces after
    text = text.rjust(len(text)+(length-len(text))) # add remaining spaces before
    return text


def show_preds(preds: list[torch.Tensor], batch_sample: list[dict], titles: list[str], vmin=0, vmax=11, show=True):
    # init titles
    titles = titles if titles is not None else [f"{i}" for i,_ in enumerate(preds)]
    # retrieve batch size
    batch_size = len(batch_sample)
    # retrieve number of predictions of different models to show
    n_preds = len(preds)
    # define figure and its size
    fig = plt.figure(figsize=(12+6*n_preds,6*n_preds))
    # first 2-rows plotting
    for i, sample in enumerate(batch_sample):
        # copy sample
        sample = {"image":sample["image"], "label":sample["label"], "name":sample["name"]}
        # in-place processing
        sample = UnNormalize()(ToNumpy()(sample))
        # retrive images
        name = sample["name"]
        img = sample["image"]
        label = sample["label"]
        # plot input image
        plt.subplot(2+n_preds, batch_size, i+1)
        plt.imshow(img, vmin=0, vmax=255)
        plt.title(f"{os.path.basename(name)}", fontsize=10+2*n_preds)
        # plot labels
        plt.subplot(2+n_preds, batch_size, batch_size+(i+1))
        plt.imshow(label, vmin=vmin, vmax=vmax)
        if i == batch_size-1: plt.title(justify("Labels", 10+2*n_preds), rotation=-90, x=1.1, y=0, fontsize=10+2*n_preds) # vertical title
    # from 3-row onwards plotting
    for j in range(n_preds):
        for i, pred in enumerate(preds[j]):
            # plot predictions
            plt.subplot(2+n_preds, batch_size, (2+j)*batch_size+(i+1))
            plt.imshow(pred, vmin=vmin, vmax=vmax)
            if i == batch_size-1: plt.title(justify(titles[j], 10+2*n_preds), rotation=-90, x=1.1, y=0, fontsize=10+2*n_preds) # vertical title
    plt.tight_layout()
    if show: plt.show()
    return fig

import torch.nn.functional as F

def contMAV_ss_score(pred_ss, mavs, vars):
    # let's build the gaussian model
    d_pred = (
        pred_ss[:, None, ...] - mavs[None, :, :, None, None]
    )  # [8,1,19,h,w] - [1,19,19,1,1] ## class-wise diff
    d_pred_ = d_pred / (vars[None, :, :, None, None] + 1e-8) ## class-wise division
    # using exponential kernel
    scores = torch.exp(-torch.einsum("bcfhw,bcfhw->bchw", d_pred_, d_pred) / 2)
    # get maximum in class dimension
    best = scores.max(dim=1)
    return 1 - best[0], best[1] # return (1-values, max_indexes)

def ss_postprocess_fn(pred_ss, pred_ow, mavs, vars):
    # contMAV scores
    ss_score1, similarity = contMAV_ss_score(pred_ss, mavs, vars) if (mavs != var_to_device(torch.zeros_like(mavs))).any() else (None, None)  # ss decoder score
    # order class logits by batch x pixel
    logits_ss = pred_ss.permute(0, 2, 3, 1)
    # get arg max of predicted classes
    preds = torch.argmax(logits_ss, dim=-1).detach()
    # add one to match label class numeration
    preds += 1
    similarity = similarity+1 if similarity is not None else None
    ss_score1 = ss_score1*(len(mavs)-1) if ss_score1 is not None else None
    return [preds.cpu()] + ([ss_score1.cpu()] if ss_score1 is not None else []) + ([similarity.cpu()] if similarity is not None else [])


def save_figure(fig, dir, filename):
    # create path
    if not os.path.exists(dir): 
        os.makedirs(os.path.abspath(dir))
    # create path
    path = os.path.join(dir, filename)
    # save figure
    fig.savefig(path)
    return path


def plot_to_image(fig, dir, filename, show=False):
    path = None
    # save figure
    if dir: 
        path = save_figure(fig, dir, filename)
    # avoid showing figure unintentionally - useful to keep RAM occupancy low
    if not show: plt.close(fig)
    # plot to image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img