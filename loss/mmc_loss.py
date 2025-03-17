import torch
from torch import nn



def mmc_loss(logits, mask, logits_mask=None):
    """
        MMC Loss (Multi-Matching Contrastive loss)
        Inspired by SupCon Loss: https://github.com/google-research/google-research/tree/master/supcon
        MMC extends it into (1) multi-modal setting and (2) batched contrastive process
        Args:
            logits: torch.Tensor[B, C], B - mini-batch size, C - number of classes
            mask: torch.Tensor[B, C], binary mask, 1 if the class is present in the image, 0 otherwise
            logits_mask: torch.Tensor[B, C], mask out self-matching logits, not applied in multi-modal setting
        Returns:
            loss_cl: torch.Tensor[1], mean cross-entropy loss over positive pairs
    """
    # flatten the batch dimension
    logits = logits.reshape(-1)
    mask = mask.reshape(-1)
    
    # for numerical stability
    logits_max = torch.max(logits)
    logits = logits - logits_max.detach()
    exp_mixed_logits = torch.exp(logits)

    # mask out self-matching logits
    if logits_mask is not None:
        logits_mask = logits_mask.reshape(-1)
        exp_mixed_logits = exp_mixed_logits * logits_mask

    # cross entropy + softmax
    log_prob = logits - torch.log(exp_mixed_logits.sum())  
    num_pos_pairs = mask.sum()

    # sum over positive pairs, division is outside the log
    num_pos_pairs = torch.where(num_pos_pairs < 1e-6, 1, num_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum() / num_pos_pairs    
    
    # mean over batch samples
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

