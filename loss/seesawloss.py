import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union



class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=2):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, y_pred, y_labels): 
        y_pred = torch.softmax(y_pred, dim=1)
        class_mask = F.one_hot(y_labels, num_classes=self.num_classes)
        pt = (y_pred * class_mask).sum(dim=1) 
        if self.alpha is None:
            loss = -((1 - pt) ** self.gamma) * pt.log()
            loss = loss.mean()
        else:
            alpha = self.alpha[y_labels]
            loss = -alpha * ((1 - pt) ** self.gamma) * pt.log()
            loss = loss.sum() / alpha.sum()
        return loss
    

class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class. 
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = F.one_hot(targets, self.num_labels)
        # print(targets.shape)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class DistibutionAgnosticSeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    p: Parameter for Mitigation Factor,
       Set to 0.8 for default following the paper.
    q: Parameter for Compensation Factor
       Set to 2 for default following the paper.
    num_labels: Class nums
    """

    def __init__(self, p: float = 0.8, q: float = 2, num_labels=2, data_name='PETA'): #num_labels=2
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.q = q
        self.class_counts = None
        self.num_labels = num_labels
        self.data_name = data_name

    def forward(self, logits, targets):
        # Mitigation Factor
        if self.class_counts is None:
            self.class_counts = (targets.sum(axis=0) + 1).float() # to prevent devided by zero.
        else:
            self.class_counts += targets.sum(axis=0)

        m_conditions = self.class_counts[:, None] > self.class_counts[None, :]
        m_trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
        m_falses = torch.ones(len(self.class_counts), len(self.class_counts)).to(targets.device)
        m = torch.where(m_conditions, m_trues, m_falses)   # [num_labels, num_labels]

        # Compensation Factor
        probility = logits

        c_condition = probility / (probility * targets).sum(dim=-1)[:, None]
        c_condition = torch.stack([c_condition] * targets.shape[-1], dim=1)
        c_condition = c_condition * targets[:, :, None]
        false = torch.ones(c_condition.shape).to(targets.device)
        c = torch.where(c_condition>1, c_condition ** self.q, false)
        
        # Sij = Mij * Cij
        s = m[None, :, :] * c
        # s = c

        # softmax trick to prevent overflow (like logsumexp trick)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow
        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)

        #seesaw loss
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

