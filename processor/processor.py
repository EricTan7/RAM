
import logging
import os
import time
import wandb
import datetime
import numpy as np
from clip import convert_weights
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
from torch.autograd import Variable

from utils.model_utils import thread_flag
from utils.model_utils import ModelEma
from utils.meter import AverageMeter
from utils.metrics import multilabel_evaluation

import warnings
warnings.filterwarnings("ignore")




def do_train(
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
        train_sampler=None,
    ):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("RAM.train")
    logger.info('start training')

    if device:
        model.to(cfg.LOCAL_RANK)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = amp.GradScaler()

    # Training
    label_smoothing = cfg.SOLVER.LABEL_SMOOTHING
    torch.cuda.empty_cache()
    thresh = cfg.SOLVER.THRESH
    ema_m = None
    tot_iters = len(train_loader) * epochs

    for epoch in range(1, epochs + 1):
        if cfg.MODEL.USE_EMA:
            if cfg.MODEL.DIST_TRAIN:
                ema_m = ModelEma(model.module, cfg.MODEL.EMA_DECAY, device=dist.get_rank())
            else:
                ema_m = ModelEma(model, cfg.MODEL.EMA_DECAY, device=device)
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        loss_meter.reset()
        acc_meter.reset()
        
        scheduler.step(epoch)
        scheduler_sgd.step(epoch)

        model.train()
        for n_iter, (img, label) in enumerate(train_loader):

            start = time.time()
            if cfg.SOLVER.LR_SCHEDULER == 'onecycle':
                scheduler.step()
                scheduler_sgd.step(epoch)
            correct = 0
            total = 0
            optimizer.zero_grad()
            optimizer_sgd.zero_grad()
            img = img.to(device)

            # construct GT matrix
            if label_smoothing:
                label_f = label.float()
                label_soft = torch.where(label_f == 1, torch.tensor(0.9), label_f)
                label_soft = torch.where(label_soft == 0, torch.tensor(0.1), label_soft)
                target = label_soft.to(device)
            else:
                target = label.to(device)

            with amp.autocast(enabled=True):
                output = model(img, target=target)

            score = output["score"]
            loss = output["loss"]

            # score, loss        
            scaler.scale(loss).backward()
            gpu_mem = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
            scaler.step(optimizer)
            scaler.step(optimizer_sgd)
            scaler.update()
            if ema_m is not None:
                if cfg.MODEL.DIST_TRAIN:
                    ema_m.update(model.module)
                else:
                    ema_m.update(model)

            targets = Variable(label)
            label = label.numpy()
            outputs_np = score.data.cpu().numpy()
            predicted = outputs_np > thresh
            correct += np.sum(predicted == label, axis=0)
            total += targets.size(0)
            acc = np.mean(correct / total)

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            torch.cuda.synchronize()

            batch_time.update(time.time() - start)

            # Logging
            if (n_iter + 1) % log_period == 0:
                if thread_flag(cfg.MODEL.DIST_TRAIN):
                    now_iter = (epoch-1) * len(train_loader) + n_iter
                    nb_remain = tot_iters - now_iter
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    cur_lr = optimizer.param_groups[0]['lr']

                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, lr: {:.2e}, mem: {:.2f}MB, speed:{:.2f}[img/s], ETA: {}"
                        .format(epoch, n_iter+1, len(train_loader), loss_meter.avg, acc, cur_lr, gpu_mem, train_loader.batch_size/batch_time.avg, eta))

                    if cfg.WANDB:
                        wandb.log({
                            "epoch": epoch,
                            "lr": cur_lr,
                            "train loss": loss_meter.avg,
                            "train acc": acc
                        })
    
        output_path = os.path.join(output_dir, f'epoch{epoch}.pth')
        if cfg.SOLVER.SAVE_MODEL and epoch % checkpoint_period == 0:
            if thread_flag(cfg.MODEL.DIST_TRAIN):
                torch.save(model.state_dict(), output_path)
            
        # Testing
        if epoch % eval_period == 0:
            if thread_flag(cfg.MODEL.DIST_TRAIN):
                Result_k, Result_k5 = validate(cfg, val_loader, model, device, zsl=True, gzsl=False)
                Result_k_gzsl, Result_k5_gzsl = validate(cfg, val_loader_gzsl, model, device, zsl=False, gzsl=True)
                now_metric = (Result_k["OF1"] + Result_k_gzsl["OF1"]) / 2.

                if ema_m is not None:
                    Result_k_ema, Result_k5_ema = validate(cfg, val_loader, ema_m.module, device, zsl=True, gzsl=False)
                    Result_k_gzsl_ema, Result_k5_gzsl_ema = validate(cfg, val_loader_gzsl, ema_m.module, device, zsl=False, gzsl=True)

                logger.info("Validation Results - Epoch: {}, F1_avg: {:.3%}".format(epoch, now_metric))
                logger.info("ZSL:")
                logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k['OP'], Result_k['OR'], Result_k['OF1']))
                logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k5['OP'], Result_k5['OR'], Result_k5['OF1']))
                logger.info("GZSL:")
                logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k_gzsl['OP'], Result_k_gzsl['OR'], Result_k_gzsl['OF1']))
                logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k5_gzsl['OP'], Result_k5_gzsl['OR'], Result_k5_gzsl['OF1']))
                if ema_m is not None:
                    logger.info("EMA Results:")
                    logger.info("ZSL:")
                    logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k_ema['OP'], Result_k_ema['OR'], Result_k_ema['OF1']))
                    logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k5_ema['OP'], Result_k5_ema['OR'], Result_k5_ema['OF1']))
                    logger.info("GZSL:")
                    logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k_gzsl_ema['OP'], Result_k_gzsl_ema['OR'], Result_k_gzsl_ema['OF1']))
                    logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k5_gzsl_ema['OP'], Result_k5_gzsl_ema['OR'], Result_k5_gzsl_ema['OF1']))

                # 3. log wandb
                if cfg.WANDB:
                    wandb.log({
                        "F1_avg ZSL-GZSL": now_metric,
                        "OP_3": Result_k['OP'], "OR_3": Result_k['OR'], "OF1_3": Result_k['OF1'],
                        "OP_5": Result_k5['OP'], "OR_5": Result_k5['OR'], "OF1_5": Result_k5['OF1'],
                        "OP_3 GZSL": Result_k_gzsl['OP'], "OR_3 GZSL": Result_k_gzsl['OR'], "OF1_3 GZSL": Result_k_gzsl['OF1'],
                        "OP_5 GZSL": Result_k5_gzsl['OP'], "OR_5 GZSL": Result_k5_gzsl['OR'], "OF1_5 GZSL": Result_k5_gzsl['OF1'],
                    })
                    if ema_m is not None:
                        wandb.log({
                            "OP_3 ema": Result_k_ema['OP'], "OR_3 ema": Result_k_ema['OR'], "OF1_3 ema": Result_k_ema['OF1'],
                            "OP_5 ema": Result_k5_ema['OP'], "OR_5 ema": Result_k5_ema['OR'], "OF1_5 ema": Result_k5_ema['OF1'],
                            "OP_3 GZSL ema": Result_k_gzsl_ema['OP'], "OR_3 GZSL ema": Result_k_gzsl_ema['OR'], "OF1_3 GZSL ema": Result_k_gzsl_ema['OF1'],
                            "OP_5 GZSL ema": Result_k5_gzsl_ema['OP'], "OR_5 GZSL ema": Result_k5_gzsl_ema['OR'], "OF1_5 GZSL ema": Result_k5_gzsl_ema['OF1'],
                        })

                torch.cuda.empty_cache()



def validate(cfg, val_loader, model, device, zsl=True, gzsl=False):
    model.eval()
    total = 0
    batch_idx = 0
    batch_time = AverageMeter()
 
    for n_iter, (img, label) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            target = label.to(device)
            start = time.time()
            output = model(img, target=target, zsl=zsl, gzsl=gzsl)
            score = output["score"]
            label = label.numpy()
            outputs_np = score.data.cpu().numpy()
            gpu_mem = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)

            batch_time.update(time.time() - start)

            if total == 0:
                g_labels = label
                p_score = outputs_np
            else:
                g_labels = np.row_stack((g_labels, label))
                p_score = np.row_stack((p_score, outputs_np))

            total += label.shape[0]
            batch_idx += 1
    
        if (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0:
            print("mem:{:.2f}, test speed:{:.2f}".format(gpu_mem, val_loader.batch_size/batch_time.avg))

    Result_k = multilabel_evaluation(p_score, g_labels, k=3)
    Result_k5 = multilabel_evaluation(p_score, g_labels, k=5)
    torch.cuda.empty_cache()

    return Result_k, Result_k5



def parse_batch(batch):
    input, label = batch
    return input, label




