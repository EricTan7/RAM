import torch
import numpy as np
import random
import torch.distributed as dist
from clip import clip
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import os
import errno
import pickle



class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            self.module.temperature = deepcopy(model.temperature)
            if hasattr(model, "temperature_glb"):
                self.module.temperature_glb = deepcopy(model.temperature_glb)
            
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def thread_flag(dist_train):
    if not dist_train:
        return True
    else:
        return dist.get_rank() == 0


def getModelSize(model):
    param_size = 0
    param_sum = 0
    grad_param_size = 0
    grad_param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
        if param.requires_grad == True:
            grad_param_size += param.nelement() * param.element_size()
            grad_param_sum += param.nelement()
    print('total number of params:{:.3f}M'.format(param_sum / 1000 / 1000))
    print('trainable number of params:{:.3f}M ({:.5%})'.format(grad_param_sum / 1000 / 1000, grad_param_sum/param_sum))

    return (param_size, param_sum, grad_param_size)



def convert_params_to_value(params):
    if params[0] == -1:
        return [-1]    # not using
    elif params[-1] == -1:
        return list(range(params[0]))    # continuous N layers
    else:
        return params


def load_clip_to_cpu(cfg, zero_shot=False):
    backbone_name = cfg.MODEL.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if zero_shot:
        saa_layer = [12, -1] if "ViT-B" in backbone_name else [24, -1]
        saa_layer = convert_params_to_value(saa_layer)
        design_details = {
            "depth_vision": [-1],
            "depth_text": [-1],
            "SAA_layer": saa_layer
        }
        print("Build zero-shot CLIP Model")
    else:
        depth_vision = convert_params_to_value(cfg.MODEL.DEPTH_VISION)
        depth_text = convert_params_to_value(cfg.MODEL.DEPTH_TEXT)
        saa_layer = convert_params_to_value(cfg.MODEL.SAA_LAYER)
        design_details = {
            "depth_vision": depth_vision,
            "vision_adapt": cfg.MODEL.VISION_ADAPT,
            "depth_text": depth_text,
            "text_ctx": cfg.MODEL.TEXT_CTX, 
            "SAA_layer": saa_layer,
            "kernel_size": cfg.MODEL.KERNEL_SIZE
        }
        print("Build CLIP Model")
    
    model = clip.build_model(state_dict or model.state_dict(), cfg.INPUT.SIZE_TRAIN, design_details)
    model.visual.SAA_replace()

    return model.float()



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    iters = state["iters"]
    if is_best:
        model_name = "model-best.pth"
    else:
        model_name = f"model-iters{iters}.pth"
    fpath = os.path.join(save_dir, model_name)

    torch.save(state, fpath)


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        fpath = 'log/my_model/model.pth.tar-10'
        checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint



def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        # >>> weight_path = 'log/my_model/model-best.pth.tar'
        # >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )
    