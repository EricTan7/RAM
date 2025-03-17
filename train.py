import os
from utils.logger import setup_logger
from datasets import make_dataloader
from model import build_model
from solver import make_optimizer, make_scheduler
from processor import do_train
from utils.model_utils import set_seed, load_clip_to_cpu, thread_flag
import torch
import torch.distributed as dist
import argparse 
from config import cfg
import warnings
warnings.filterwarnings("ignore")
import time
import wandb



def main(cfg):
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(cfg.LOCAL_RANK)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()

    # 1. Logging
    if cfg.WANDB:
        run = wandb.init(project=cfg.WANDB_PROJ, config=cfg)
        run.name = f'{cfg.DATASETS.NAMES}-{cfg.SOLVER.OPTIMIZER_NAME}-lr{cfg.SOLVER.BASE_LR}'

    output_dir = os.path.join(cfg.OUTPUT_DIR, 'RAM-' + time.strftime('%Y-%m-%d-%H-%M-%S'))
    if thread_flag(cfg.MODEL.DIST_TRAIN):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = setup_logger("RAM", output_dir, if_train=True)
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info("Running with config:\n{}".format(cfg))

    # 2. Data
    train_loader, val_loader, val_loader_gzsl, train_sampler, dataset = make_dataloader(cfg)

    # 3. Model
    clip_model = load_clip_to_cpu(cfg)
    clip_model_teacher = load_clip_to_cpu(cfg, zero_shot=True).eval()
    model = build_model(cfg, clip_model, dataset, clip_model_teacher)


    if cfg.MODEL.LOAD:
        state_dict = torch.load(cfg.TEST.WEIGHT)
        model.load_state_dict(state_dict)
        print(f"Load weights from: {cfg.TEST.WEIGHT}")

    # 4. Optimizer
    optimizer, optimizer_sgd = make_optimizer(cfg, model)
    scheduler, scheduler_sgd = make_scheduler(cfg, optimizer, len(train_loader)), make_scheduler(cfg, optimizer_sgd, len(train_loader))

    # 5. Start training
    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        val_loader_gzsl,
        optimizer,
        optimizer_sgd,
        scheduler,
        scheduler_sgd,
        output_dir,
        train_sampler,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAM Training")
    parser.add_argument(
        "--config_file",
        help="path to config file", type=str, default="configs/coco.yml"
    )
    parser.add_argument(
        "--local_rank",
        type=int, default=0
    )
    parser.add_argument(
        "opts", 
        help="Modify config options from command-line", nargs=argparse.REMAINDER, default=None
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.LOCAL_RANK = args.local_rank

    main(cfg)



