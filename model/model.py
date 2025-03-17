import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import clip
from clip import clip
from .base import BaseModel
from .ot_solver import Sinkhorn
import ipdb


class RAM_Model(BaseModel):
    """
        Entropic KL-regularized Unbalanced Optimal Transport (EUOT)
        SupCon Loss
    """
    def __init__(
            self,
            cfg,
            clip_model,
            classnames_seen,
            classnames_unseen,
            clip_model_teacher
    ):
        super().__init__()
        self.cfg = cfg
        self.classnames_seen = classnames_seen
        self.classnames_unseen = classnames_unseen
        self.num_classes_seen = len(classnames_seen)
        self.num_classes_unseen = len(classnames_unseen)
        self.criterion = self.make_criterion(cfg)
        self.device = dist.get_rank() if cfg.MODEL.DIST_TRAIN else 'cuda'

        # create teacher model, preserve teacher text features
        self.clip_model = clip_model
        self.text_tokens = self.get_text_templates()
        self.text_fea_teacher = self.get_text_fea(clip_model_teacher, classnames_seen+classnames_unseen)
        self.clip_model_teacher = clip_model_teacher

        # freeze the model
        self.freeze(cfg.MODEL.TRANSFER_TYPE)

        # pre-trained logit scale
        self.logit_scale = self.clip_model.logit_scale.exp()
        # learnable temperature
        self.temperature_loc = nn.Parameter(torch.tensor(cfg.MODEL.TEMPERATURE))
        self.temperature = nn.Parameter(1./self.logit_scale)
        
        # KCOT parameters
        self.reg = cfg.MODEL.OT_REG
        self.reg_sc = cfg.MODEL.OT_REGSC

    def get_text_fea(self, clip_model, classnames):
        text_templates = "A photo of a {}."
        text_templates = [text_templates.format(classnames[i]) for i in range(len(classnames))]
        text_tok = clip.tokenize(text_templates)
        with torch.no_grad():
            text_fea = clip_model.encode_text(text_tok)
        return text_fea.unsqueeze(1).detach()
    
    def get_text_templates(self):
        templates = "A photo of a {}."
        texts = [templates.format(name) for name in self.classnames_seen+self.classnames_unseen]
        text_tokens = clip.tokenize(texts)
        return text_tokens.cuda()
    
    def build_weights(self, sim, dim=-1, temperature=0.1):
        with torch.no_grad():
            sim_max = sim.max(dim=dim)[0]
            weights = (sim_max / temperature).softmax(dim=-1)
        return weights
    
    def generate_teacher_distribution(self, img_teacher, zsl=False, gzsl=False):
        with torch.no_grad():
            _, img_loc = self.clip_model_teacher.visual(img_teacher)
            img_loc = img_loc[0][:, 1:]
            text_fea = self.text_fea_teacher.clone().cuda()
            if zsl:
                text_fea = text_fea[self.num_classes_seen:]
            elif gzsl:
                pass
            else:
                text_fea = text_fea[:self.num_classes_seen]
            B, tok, dim=img_loc.shape
            C, gp, dim = text_fea.shape

            text_fea = F.normalize(text_fea, dim=-1) 
            img_loc = F.normalize(img_loc, dim=-1)
            text_fea = text_fea.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, dim)

            # generate weight
            logit_scale = self.clip_model_teacher.logit_scale.exp()
            logits_loc = logit_scale * img_loc @ text_fea.transpose(-2, -1) 
            logits_loc = logits_loc.reshape(B, -1, C, gp)
            local_similarity = logits_loc.softmax(dim=2)
            prob = (local_similarity*20.).softmax(dim=1)
            prob = prob.mean(dim=-1)
        return prob

    def forward(self, img, target=None, zsl=False, gzsl=False):
        seen = True if not zsl and not gzsl else False
        if seen:
            text_tokens = self.text_tokens[:self.num_classes_seen].clone()
        elif zsl:
            text_tokens = self.text_tokens[self.num_classes_seen:self.num_classes_seen+self.num_classes_unseen].clone()
        else:
            text_tokens = self.text_tokens[:self.num_classes_seen+self.num_classes_unseen].clone()
        prompt_fea_loc = self.clip_model.encode_text(text_tokens)
        prompt_fea_loc = prompt_fea_loc.unsqueeze(1)
        
        img_glb, img_loc = self.clip_model.visual(img, text_fea=prompt_fea_loc)
        img_loc = img_loc[0][:, 1:]

        B, tok, dim = img_loc.shape
        C, gp, dim = prompt_fea_loc.shape

        prompt_fea_loc = prompt_fea_loc.permute(1, 0, 2)

        img_glb = F.normalize(img_glb, dim=-1)
        img_loc = F.normalize(img_loc, dim=-1)
        prompt_fea_loc = F.normalize(prompt_fea_loc, dim=-1)

        logits_glb = img_glb @ prompt_fea_loc.transpose(1, 2) / self.temperature
        score_glb = logits_glb.squeeze(1).softmax(dim=-1)
        if self.training:
            mask = target
            loss_glb = self.criterion(logits_glb, mask)
        
        # Cost matrix
        sim = img_loc @ prompt_fea_loc.transpose(1, 2)
        cost = (sim * self.logit_scale).softmax(dim=-1)
        cost = 1.0 - cost

        if self.training:
            # Teacher is only applied in training
            frozen_mask = self.generate_teacher_distribution(img, zsl, gzsl)
            gt_mask = target.unsqueeze(1).expand(-1, tok, -1)
            frozen_mask[gt_mask==0] = frozen_mask.min()
            cost_tr = -torch.log(frozen_mask) * self.reg_sc
            cost = cost + cost_tr
            reg = self.reg + self.reg_sc
        else:
            reg = self.reg

        u = self.build_weights(sim.detach(), dim=2, temperature=0.1)
        v = torch.zeros((B, C), dtype=sim.dtype, device=sim.device).fill_(1./C)
        with torch.no_grad():
            T = Sinkhorn(u, v, cost, reg=reg)
        if torch.isnan(T).any():
            raise ValueError("Found nan in OT matrix!")
        
        sim_op = T * sim
        sim_op = sim_op.sum(dim=1) / self.temperature_loc
        score_loc = sim_op.softmax(dim=-1)
        score = (score_glb + score_loc) / 2.
        if self.training:
            mask = target
            loss_loc = self.criterion(sim_op, mask)
            loss = loss_glb + loss_loc
            return {"score": score, "loss": loss}
        else:
            return {"score": score}



