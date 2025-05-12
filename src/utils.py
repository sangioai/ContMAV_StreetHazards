########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import sys

import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

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
        # if epoch: # don't need this
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
                # logging(f"label_sq:{label_sq}")
                oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                # logging(f"oh_label:{oh_label}")
                count = oh_label.view(-1, oh_label.shape[-1]).sum(
                    dim=0
                )  # num of pixels per cl
                # logging(f"count:{count}")
                pred = emb.permute(1, 2, 0).unsqueeze(-1)
                # logging(f"emb.shape:{emb.shape}")
                # logging(f"pred.shape:{pred.shape}")
                oh_pred = (
                    pred * oh_label
                )  # (H, W, Nc, Ncp) Ncp num classes present in the label
                
                # logging(f"oh_pred.shape:{oh_pred.shape}")
                oh_pred_flatten = oh_pred.view(
                    oh_pred.shape[0] * oh_pred.shape[1],
                    oh_pred.shape[2],
                    oh_pred.shape[3],
                )
                res_raw = oh_pred_flatten.sum(dim=0) / count  # avg feat per class
                res_new = (res_raw[~res_raw.isnan()]).view(
                    -1, self.n_classes
                )  
                # logging(f"res_new:{res_new}")
                # logging(f"emb_k:{emb_k}")
                # filter out nans given by intermediate classes (present because of oh)
                label_list = label.unique()
                if self.n_classes in label_list:
                    label_list = label_list[:-1]
                    res_new = res_new[:-1, :]

                # temperature-scaled cosine similarity
                final = (var_to_device(res_new) @ var_to_device(emb_k.T)) / tau

                # logging(f"final:{final}")
                # logging(f"label:{label}")
                # logging(f"label_list:{label_list}")
                loss = F.cross_entropy(final, label_list)
                total_loss += loss

        return total_loss / emb_q.shape[0]

        return var_to_device(torch.tensor(0))


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
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        # self.ex = {i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)}
        # self.ex2 = {
        #     i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
        # }

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_features2 = None
        self.previous_count = None


    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        # logging(f"|---> OWLoss cumulate")
        import pandas as pd
        # logging(f"input logits:{logits.shape}")
        # logging(pd.DataFrame(logits.detach().cpu().flatten()).describe())
        # logging(f"input sem_gt:{sem_gt.shape}")
        # logging(pd.DataFrame(sem_gt.detach().cpu().flatten()).describe())
        # pixel wise prediction w/ softmax
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1) 
        # logging(f"sem_pred.shape:{sem_pred.shape}")
        # logging(pd.DataFrame(sem_pred.detach().cpu().flatten()).describe())
        # list of labels present in this ground-truth target tensor
        gt_labels = torch.unique(sem_gt).tolist()
        # logging(f"gt_labels:{gt_labels}")
        # let's order by pixels
        logits_permuted = logits.permute(0, 2, 3, 1)
        # for all label classes in this gt target tensor
        for label in gt_labels:
            # logging(f"---->    label:{label}")
            # if anomaly/void/unlabeled - skip
            if label == 255:
                continue
            # label mask on gt target
            sem_gt_current = sem_gt == label
            # label mask ok prediction (softmax tensor)
            sem_pred_current = sem_pred == label
            # true-positive mask btw this label and predictions
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            # logging(f"tps_current:{tps_current.shape}")
            # logging(pd.DataFrame(tps_current.detach().cpu().flatten()).describe())
            # skip if no true-positive available
            if tps_current.sum() == 0:
                continue
            # get logtits where true-positives
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # logging(f"logits_tps.shape:{logits_tps.shape}")
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            # get mean of true pos logits
            avg_mav = torch.mean(logits_tps, dim=0)
            # logging(f"avg_mav:{avg_mav.shape}")
            # logging(pd.DataFrame(avg_mav.detach().cpu().flatten()).describe())
            # get number of true positives
            n_tps = logits_tps.shape[0]
            # logging(f"n_tps:{n_tps}")
            # # features is running mean for mav
            # # logging(f"self.features[{label}]-0:{self.features[label]}")
            # self.features[label] = (
            #     self.features[label] * self.count[label] + avg_mav * n_tps
            # )
            # # logging(f"self.features[{label}]-1:{self.features[label]}")
            # # save sum of tp logits
            # self.ex[label] += (logits_tps).sum(dim=0)
            # # logging(f" self.ex[{label}]:{ self.ex[label]}")
            # # save sum of squared tp logits
            # self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            # # logging(f" self.ex2[{label}]:{ self.ex2[label]}")
            # # cumulate number of tp found
            # self.count[label] += n_tps
            # # logging(f"self.count[{label}]:{self.count[label]}")
            # # 
            # self.features[label] /= self.count[label] + 1e-8
            # # logging(f"self.features[{label}]-2:{self.features[label]}")
            ####
            # my version - welford alg https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            ###
            # cumulate number of tp found
            self.count[label] += n_tps
            # logging(f"self.count[{label}]: {self.count[label]} (here:{n_tps})")
            # # sum pixels feature-wise 
            # logits_tps_sum = (logits_tps).sum(dim=0)
            # # logging(f"logits_tps_sum: {logits_tps_sum.shape} {logits_tps_sum}")
            #-> but can be large, let's divide before instead of later (-> compute contributions before then reduce)
            logits_tps_contrib_sum = (logits_tps / (self.count[label] + 1e-8)).sum(dim=0)
            # logging(f"logits_tps_contrib_sum: {logits_tps_contrib_sum.shape} {logits_tps_contrib_sum}")
            #-> let's calc features contributions as well
            features_contrib = self.features[label] / (self.count[label] + 1e-8)
            # logging(f"features_contrib: {features_contrib.shape} {features_contrib}")
            # get new data delta from mean
            delta = logits_tps_contrib_sum - features_contrib
            # logging(f"delta:{delta}")
            # get new mean for label - add new contributions
            self.features[label] = self.features[label] + delta # instead of: self.features[label] + (delta \ count)
            # logging(f"self.features[label]:{self.features[label]}")
            #-> let's calc features contributions as well
            features_contrib = self.features[label] / (self.count[label] + 1e-8)
            # logging(f"features_contrib: {features_contrib.shape} {features_contrib}")
            # get new delta from new mean
            delta2 = logits_tps_contrib_sum - features_contrib
            # accumulate delta differences
            self.features2[label] += (delta * delta2)
            
            # import pandas as pd
            # print(f"label=[{label}]")
            # print(pd.DataFrame(logits_tps.detach().cpu().flatten()).describe())
            # logging(f"delta2:{delta2}")
            # #-> let's calc features2 contributions as well
            # var_contrib = self.var[label] / (self.count[label] + 1e-8)
            # # logging(f"var_contrib: {var_contrib.shape} {var_contrib}")
            # get new squared mean - useful for finalize standard deviation
            # # logging(f"self.features2[label]:{self.features2[label]}")
            # # get new variance for label
            # self.var[label] = self.features2[label] / self.count[label]
            # logging(f"self.var[label]:{self.var[label]}")
            
            # print(f"self.count[{label}]: {self.count[label]} (here:{n_tps}) (mean.mean:{self.features[label].mean()} (var.mean:{self.var[label].mean()})")

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

        # acc_loss = var_to_device(torch.tensor(0.0))
        # for label in gt_labels[:-1]:
        #     mav = self.previous_features[label]
        #     logs = logits_permuted[torch.where(sem_gt == label)]
        #     mav = mav.expand(logs.shape[0], -1)
        #     if self.previous_count[label] > 0:
        #         ew_l1 = self.criterion(logs, mav)
        #         ew_l1 = ew_l1 / (self.var[label] + 1e-8)
        #         if self.hinged:
        #             ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
        #         acc_loss += ew_l1.mean()

        acc_loss = var_to_device(torch.tensor(0.0))
        
        for label in gt_labels[:-1]:
            # finalize accumulations
            mav = self.previous_features[label]
            var =  self.previous_features2[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                num = (self.criterion(logs, mav) ** 2 ) if self.mav_squared else (self.criterion(logs, mav))
                den = (var[label] ** 2 + 1e-8)     if self.mav_squared else (var[label] + 1e-8)
                # print(f"mav_squared[{label}]:{self.mav_squared}")
                # print(f"num.max():{num.max()} num.min():{num.min()}")
                # print(f"den.max():{den.max()} den.min():{den.min()}")
                ew_l1 = num / den # squared
                # logging(f"self.features[{label}]:{self.features[label]}")
                # logging(f"self.var[{label}]:{self.var[label]}")
                # logging(f"ew_l1[{label}]-1:{ew_l1}")
                # logging(f"ew_l1[{label}]-2:{ew_l1}")
                ew_l1_mean = ew_l1.mean()
                # logging(f"ew_l1[{label}].mean-3:{ew_l1_mean}")
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                    # logging(f"hinged ew_l1[{label}]:{ew_l1}")
                acc_loss += ew_l1_mean # instead of mean
                # logging(f"acc_loss:{acc_loss}")
        
        # print(f"count.max():{self.previous_count.max()} den.min():{self.previous_count.min()}")
        return acc_loss

    def update(self):
        # self.previous_features = self.features
        # self.previous_count = self.count
        # for c in self.var.keys():
        #     self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (
        #         self.count[c] + 1e-8
        #     )

        # # resetting for next epoch
        # self.count = torch.zeros(self.n_classes)  # count for class
        # self.features = {
        #     i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
        # }
        # self.ex = {i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)}
        # self.ex2 = {
        #     i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
        # }

        # return self.previous_features, self.var
        if self.previous_features is not None:
            return self.previous_features, self.previous_features2
        zeros = {
            i: var_to_device(torch.zeros(self.n_classes)) for i in range(self.n_classes)
            }
        return zeros, zeros
    
    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        # if self.previous_features is not None:
        #     for key in self.previous_features.keys():
        #         mav_tensor[key] = self.previous_features[key]
        
        if self.features is not None:
            for key in self.features.keys():
                mav_tensor[key] = self.features[key]
        return mav_tensor


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
        # logging(f"input inputs.shape:{inputs.shape} targets.shape:{targets.shape}")
        losses = []
        targets_m = targets.clone()
        if targets_m.sum() == 0:
            import ipdb;ipdb.set_trace()  # fmt: skip
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(
            targets.flatten().type(self.dtype), minlength=self.num_classes
        )
        import pandas as pd
        # logging(f"targets values range: {targets.min()} - {targets.max()}")

        # logging(f"after bincount number_of_pixels_per_class.shape:{number_of_pixels_per_class.shape}")
        # logging(f"remember self.weight.shape:{self.weight.shape} values:{self.weight}")
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
