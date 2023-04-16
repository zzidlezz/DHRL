import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

# Check in 2022-1-3
from torch import nn
from torch.autograd import Variable

def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def Ranking(y, p, alpha, beta):

    # Invert labels of batch
    inverted_y = torch.add(torch.mul(y, -1.), 1.)

    # I only want to do pairwise computation between 0 and 1 labels. This eliminates the 1-1 and 0-0 pairs.
    select_examples = torch.mul(y.unsqueeze(1), inverted_y.unsqueeze(2))


    # Calculate the errors
    k = torch.sub(p.unsqueeze(2), p.unsqueeze(1))
    k = torch.mul(k, 2.)
    k = torch.add(k, 1.)
    k = torch.maximum(torch.tensor(0), k)
    errors = torch.square(k)

    # Select instances: Zero the 1-1 and 0-0 pairs out
    errors = torch.mul(errors, select_examples)

    # Calculate the loss_miss
    k = torch.sum(errors, 2)
    # Escape instability when taking the gradient of any 0 values with tf.math.sqrt
    k = k + 1e-10
    groups_miss = torch.sqrt(k)
    loss_miss = torch.sum(groups_miss,1)

    # Choose the potential missing class label for every sample
    potential_missing = torch.argmax(groups_miss, 1)
    missing_high = torch.max(groups_miss, 1)[0]

    # Calculate the loss_extra
    k = torch.sum(errors, 1)
    # Escape instability when taking the gradient of any 0 values with tf.math.sqrt
    k = k + 1e-10
    groups_extra = torch.sqrt(k)
    loss_extra = torch.sum(groups_extra, dim = 1)

    # Choose the potential extra label for every sample
    potential_extra = torch.argmax(groups_extra, dim = 1)
    extra_high = torch.max(groups_extra, dim = 1)[0]

    # Choose the missing or the extra per sample
    classes = torch.where(missing_high > extra_high, potential_missing, potential_extra)

    # Enter the loss for the sample
    losses = torch.add(torch.mul(alpha, loss_miss), torch.mul(beta, loss_extra))

    return losses, classes


class LRLoss(nn.Module):

    def __init__(self, pos_weight):
        super(LRLoss, self).__init__()
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if not (targets.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))
        max_val = (-logits).clamp(min=0)
        log_weight = 1 + (pos_weight - 1) * targets
        loss = (1 - targets) * logits + log_weight * (((-max_val).exp() + (-logits - max_val).exp()).log() + max_val)

        return loss.sum(axis=1)

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap