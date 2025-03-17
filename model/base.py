
import os
import torch
import torch.nn as nn
from collections import OrderedDict

from utils.model_utils import save_checkpoint, load_checkpoint, tolist_if_not
from loss import AsymmetricLossOptimized, AsymmetricLossOptimized_partial, mmc_loss


class BaseModel(nn.Module):
    """
    Basic Model
    Implementation of some common functions
    """
    def __init__(self):
        super().__init__()
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def register_model(self, name="model", model=None, optim=None, sched=None):
        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]

    def get_specific_lr(self, names=None):
        if names is None:
            names = self.get_model_names(names)
            name = names[0]
        else:
            name = names
        return self._optims[name].param_groups[0]["lr"]

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def save_model(self, iters, directory, is_best=False):
        # save registered_module
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()
            save_dict = OrderedDict()
            for k, v in self._models[name].named_parameters():
                if v.requires_grad:
                    save_dict[k] = model_dict[k]

            sdir = os.path.join(directory, name)
            save_checkpoint(
                {
                    "state_dict": save_dict,
                    "iters": iters,
                },
                sdir,
                is_best
            )
            
            print(f"Checkpoint of {name} saved to {sdir}")

    def load_model(self, directory, iters):
        model_file = f"model-iters{iters}.pth"
        names = self.get_model_names()

        for name in names:
            model_path = os.path.join(directory, name, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            iters = checkpoint["iters"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}"'.format(name, model_path))
            self._models[name].load_state_dict(state_dict, strict=False)

    def make_criterion(self, cfg):
        """
            Classification loss
                - Zero-shot setting: MMC loss
                - Partial-label setting: ASL partial loss
        """
        if cfg.MODEL.LOSS_TYPE == 'MMC':
            criterion = mmc_loss
        elif cfg.MODEL.LOSS_TYPE == 'ASL':
            criterion = AsymmetricLossOptimized(cfg.SOLVER.GAMMA_NEG, cfg.SOLVER.GAMMA_POS, cfg.SOLVER.CLIP)
        elif cfg.MODEL.LOSS_TYPE == 'ASL-partial':
            criterion = AsymmetricLossOptimized_partial(cfg.SOLVER.GAMMA_NEG, cfg.SOLVER.GAMMA_POS, cfg.SOLVER.CLIP)
        else:
            raise NotImplementedError

        return criterion

    def freeze(self, transfer_type):
        if hasattr(self, "clip_model_teacher") and self.clip_model_teacher is not None:
            for name, param in self.clip_model_teacher.named_parameters():
                param.requires_grad = False

        if transfer_type == "no_freeze":
            pass
        
        elif transfer_type == "freeze_all":
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
        
        elif transfer_type == "freeze_text":
            for name, param in self.clip_model.named_parameters():
                if 'visual.' in name:
                    continue
                else:
                    param.requires_grad = False

        elif transfer_type == "Adapter":
            for name, param in self.clip_model.named_parameters():
                if "adapter" in name or "embeds" in name:      #  embeds
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif "partial" in transfer_type:
            total_layer = len(self.clip_model.visual.transformer.resblocks)
            partial_layer = int(transfer_type.split("-")[-1])
            if partial_layer > total_layer:
                raise NotImplementedError
            for name, param in self.clip_model.named_parameters():
                find = False
                for l in range(total_layer-partial_layer, total_layer):
                    if "visual.transformer.resblocks.{}".format(l) in name:
                        param.requires_grad = True
                        find = True
                        break
                if not find:
                    param.requires_grad = False
                
        else:
            raise NotImplementedError

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


