from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# CLIP Backbone
_C.MODEL.BACKBONE = 'ViT-B/16'
# Use BN
_C.MODEL.BN = False
# Number of head
_C.MODEL.NUM_HEAD = 8
# Loss type
_C.MODEL.LOSS_TYPE = 'BCE'  # 'ASL'
# ema model
_C.MODEL.USE_EMA = False
_C.MODEL.EMA_DECAY = 0.9997
# load pretrain 
_C.MODEL.LOAD = False
# text encoder
_C.MODEL.TEXT_ENCODER = 'CLIP'
_C.MODEL.DEPTH_TEXT = [-1]
_C.MODEL.TEXT_CTX = 4     # middle layers, 一般不用
_C.MODEL.PROMPT_CSC = False

# transfer type
_C.MODEL.TRANSFER_TYPE = "freeze_all"

# SAA
_C.MODEL.SAA_LAYER = [-1]

# loc region pooling
_C.MODEL.LOC_STRIDE_SIZE = 4
_C.MODEL.LOC_KERNEL_SIZE = 4

# Adapter
_C.MODEL.DEPTH_VISION = [-1]
_C.MODEL.VISION_ADAPT = 8
_C.MODEL.KERNEL_SIZE = 3

# temperature
_C.MODEL.TEMPERATURE = 0.002

# new prototype
_C.MODEL.NUM_NEW_PROTOTYPE = 10

# OT reg
_C.MODEL.OT_REG = 0.1
_C.MODEL.OT_REGSC = 0.05

# Adapter ratio
_C.MODEL.ADAPTER_RATIO = 0.2


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [224, 224]
# Size of the image during test
_C.INPUT.SIZE_TEST = [224, 224]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
# _C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# # Values to be used for image normalization
# _C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
# Value of padding size
_C.INPUT.PADDING = 10

# Image augmentation
_C.INPUT.HIDESEEK = False
_C.INPUT.AUGMIX = False

# Vis: with path dataset
_C.INPUT.WITH_PATH = False

# Text template
_C.INPUT.TEMPLATE = 'vanilla'   # 'ensemble'
_C.INPUT.ENSEMBLE_TYPE = 'embedding'    # 'logit' 'score'
_C.INPUT.NUM_GROUPS = 1    # 'logit'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('PETA')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# pedestrain attribute numbers
_C.DATASETS.NUMBERS = 35
_C.DATASETS.TEMPLATE_NAMES = ''

# label proportion
_C.DATASETS.PARTIAL = -1.0

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Save the model
_C.SOLVER.SAVE_MODEL = False
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Number of max epoches FOR SCHEDULER
_C.SOLVER.SCHEDULER_MAX_EPOCHS = 60
# Number of max epoches FOR SCHEDULER
_C.SOLVER.SCHEDULER_MAX_ITER = 1000000
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 42
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.LARGE_FC_LR = False

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
_C.SOLVER.WEIGHT_DECAY_SGD = 0.0001

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 3
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU 
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# Classification Threshold
# Loss type for contrastive
_C.SOLVER.THRESH = 0.5

# Label smoothing
_C.SOLVER.LABEL_SMOOTHING = False

# LR sheduler iter (TGPT imple)
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.LR_SCHEDULER = "cosine"
_C.SOLVER.STEPSIZE = 1000

# aslloss param
_C.SOLVER.GAMMA_NEG = 2
_C.SOLVER.GAMMA_POS = 0
_C.SOLVER.CLIP = 0.

# twloss param
_C.SOLVER.TP = 4.
_C.SOLVER.TN = 1.

# save the middle output, for visualization
_C.SOLVER.VERBOSE = False

# iter training
_C.SOLVER.MAX_ITER = 12800
_C.SOLVER.WARMUP_ITER = 200
_C.SOLVER.BASE_LR_SGD = 0.001

# KD loss weight
_C.SOLVER.KDLOSS_WEIGHT = 1.

# Text batch
_C.SOLVER.TEXT_BATCH_SIZE = 80

# debug mode
_C.SOLVER.DEBUG = False

# zero-shot testing
_C.SOLVER.ZS_TEST = False

# sample text
_C.SOLVER.SAMPLE_TEXT = False


# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
_C.TEST.WEIGHT_ITERS = 12800
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
_C.TEST.USE_FUSION = False
_C.TEST.TTA = -1
_C.TEST.TEN_CROP = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""


# ---------------------------------------------------------------------------- #
# Train time configs
# ---------------------------------------------------------------------------- #
_C.LOCAL_RANK = 0
_C.WANDB = False
_C.WANDB_PROJ = "OVML-RAM"






