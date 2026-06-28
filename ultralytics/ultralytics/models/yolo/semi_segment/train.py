# # ulralytics/models/yolo/semi-segment/train.py

from pathlib import Path
import os
import math
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import torch
from torch import nn, optim
from torch import distributed as dist
import torch.nn.functional as F
import torchvision
from random import random

import subprocess
from ultralytics.engine.results import Results
from copy import deepcopy
from ultralytics.models import yolo
from ultralytics.nn.tasks import SemiSegmentationModel
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import ops
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.data import build_dataloader
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.nn.autobackend import check_class_names

from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
    yaml_load,
)

from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
from ultralytics.models.yolo.semi_segment.aug_unsup import StrongNoiseBlurAug,WeekAugmentation,HorizonFlip
from ultralytics.models.yolo.semi_segment.utils import draw_instances,visualize_instance_segmentation


class SemiSegmentationTrainer(yolo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "semi_segment"
        super().__init__(cfg, overrides, _callbacks)


        self.ema_decay = 0.9
        self.unsup_weight = getattr(self.args, "unsup_weight", 1.0)  # 支持从 args 读取权重
        self.auto_train = getattr(self.args, "self_train", False)
        self.total_loss = None
        self.lamda = 0.7 #最终lamda能到0.7
        self.strong_aug = StrongNoiseBlurAug()
        self.weak_aug = WeekAugmentation()
        self.horizonFlip = HorizonFlip()
        self.reset_teacher = False
        self.update_count = 0
        # # 无标签数据加载器先不初始化，放到 setup 阶段
        # self.unsup_loader = None
        #如果有教师模型的预权重，为其单独再创造一套对应参数
        if self.args.unsup_model:
            self.teacher_model = check_model_file_from_stem(self.args.unsup_model)
        else:
            self.teacher_model = None
        self.teacher_ema = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SemiSegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)

    @torch.no_grad()
    def update_teacher_model(self):
        """Update the teacher model using EMA from student model."""

        if self.reset_teacher:#因为没有优秀模型导致重置时不进行EMA更新
            self.teacher_model = deepcopy(self.best_student)
            print("出现新的最优模型,重置一次教师模型\n")
            self.reset_teacher = False
            #self.update_count = 0
            student_model = self.model.module if hasattr(self.model, "module") else self.model
            teacher_model = self.teacher_model
            self.update_count = 0
        elif self.update_count == 30:
            print("太久未出现优秀模型，重置教师模型且本轮不再更新教师模型")
            best_student = self.best_student
            student_model = self.model.module if hasattr(self.model, "module") else self.model
            self.teacher_model = deepcopy(self.best_student)
            for t_param, s_param in zip(self.teacher_model.parameters(), best_student.parameters()):
                t_param.data = s_param.data
            teacher_model = self.teacher_model
        elif self.update_count > 30:
            self.update_count = 0
            return
        else:
            min_decay = 0.9
            max_decay = 0.99
            cos_value = math.cos(math.pi * self.epoch / self.epochs)
            self.ema_decay = max_decay - 0.5 * (max_decay - min_decay) * (1 + cos_value)

            # 从 student 模型获取参数
            student_model = self.model.module if hasattr(self.model, "module") else self.model
            teacher_model = self.teacher_model
            if self.best_student is None:
                for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                    if t_param.data.shape != s_param.data.shape:
                    # 如果参数 shape 不一致，跳过（可能是结构不兼容）
                        print("参数shape不一致,跳过同步\n")
                        continue
                    t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)
            else:
                best_model = self.best_student
                alpha = (1.0 - self.ema_decay) * 0.5
                beta = (1.0 - self.ema_decay) * 0.5
                for t_param, s_param, b_param in zip(teacher_model.parameters(), student_model.parameters(),best_model.parameters()):
                    if t_param.data.shape != s_param.data.shape:
                    # 如果参数 shape 不一致，跳过（可能是结构不兼容）
                        print("参数shape不一致,跳过同步\n")
                        continue
                    t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=alpha).add_(b_param.data, alpha=beta)

        # 同步 buffer（如 BN 层的 running_mean、running_var 等）
        for t_buffer, s_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            if t_buffer.data.shape != s_buffer.data.shape:
                continue
            t_buffer.data.copy_(s_buffer.data)

        self.teacher_model.eval()

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0
        
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                    raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _do_train(self, world_size=1):
        
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        if self.auto_train:
            nb = len(self.unsup_loader)# 每个 epoch 的 batch 数
        else:
            nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup 的总 iteration 数
        last_opt_step = -1  # warmup 的总 iteration 数
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        #训练后期关闭 Mosaic 数据增强，并标记这几个 batch 做可视化。
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        if self.teacher_optimizer is not None:
            self.teacher_optimizer.zero_grad()

        # MOD: 初始化无监督滑动平均日志（仅打印总体无监督损失，不再打印监督四项）
        self.tunsup_student = None
        self.tunsup_teacher = None

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
                self.teacher_scheduler.step()

            self.model.train()
            self.teacher_model.train()
            #self.teacher_model.eval()
            if RANK != -1:#打乱一次乱序数据的同步
                self.train_loader.sampler.set_epoch(epoch)
                self.unsup_loader.sampler.set_epoch(epoch)
            #pbar = enumerate(self.train_loader)

            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                if not self.auto_train:
                    self.train_loader.reset()
                self.unsup_loader.reset()

            if self.auto_train:
                base_iter = enumerate(self.unsup_loader)

                def wrap_unsup(iterable):
                    for i, u in iterable:
                        # 用 None 占位 batch，使循环解包统一为 (batch, unsup_batch)
                        yield i, (None, u)

                it = wrap_unsup(base_iter)
            else:
                it = enumerate(zip(self.train_loader, self.unsup_loader))

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(it, total=nb)
            else:
                pbar = it

            self.tloss = None
            self.tloss1 = None
        
            if epoch < 200:
                lamda = 0
            else:
                lamda = min((0.5 * epoch / self.epochs),0.4)

            if self.auto_train:
                lamda = 1.0
                
            for i, (batch,unsup_batch) in pbar:
                #print(f"len of batch is:{len(batch['im_file'])}\tlen of unsup_batch is:{len(unsup_batch['im_file'])}")

                self.run_callbacks("on_train_batch_start")
                #self.semi_epoch = 250

                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    if self.teacher_optimizer is not None:
                        for j, x in enumerate(self.teacher_optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])


                with autocast(self.amp):
                    if batch is not None:
                        batch = self.preprocess_batch(batch)
                    unsup_batch = self.preprocess_semi_batch(unsup_batch)

                    if not self.auto_train:
                        #step1：学生模型根据标签数据进行学习（非auto-train模式时）
                        self.loss, self.loss_items = self.model(batch)
                        self.total_loss = self.loss

                        self.loss1, self.loss_items1 = self.teacher_model(batch)
                        self.total_loss1 = self.loss1
                    else:
                        self.loss1 = self.loss = 0

                    if lamda != 0:
                    #     #教师模型初次产生时同步最佳学生
                    #     if self.first_unsup:
                    #         self.first_unsup = False
                    #         with torch.no_grad():
                    #             if self.best_student is not None:
                    #                 self.teacher_model = deepcopy(self.best_student)
                    #             else:
                    #                 self.teacher_model = deepcopy(self.model)


                        #第一次预测时同步教师模型和学生模型                        
                                # for p in self.teacher_model.parameters():
                                #     p.requires_grad = False
                                # self.teacher_model.eval()
                        #unsup_batch完全未经过数据增强
                        #复制一份deepcopy的unsup_batch，并对unsup_batch进行弱增强（水平翻转，高斯噪声）
                        unsup_w = deepcopy(unsup_batch)
                        unsup_w,horizons = self.weak_aug(unsup_w)#进行增强并记录哪些图像被翻转了
                        #教师模型预测弱增强结果
                        with torch.no_grad():

                        #step2：教师模型预测结果监督学生
                            self.model.train()
                            self.teacher_model.eval()
                            pseudo_pred, _ = self.teacher_model(unsup_w)
                            # pseudo_pred[0] = [p.detach() for p in pseudo_pred[0]]
                            # pseudo_pred[1] = [p.detach() for p in pseudo_pred[1]]
                            #将预测结果转换为伪标签
                            pseudo_box_label = self.get_pseudo_box_label(pseudo_pred[0])
                            pseudo_label = self.get_pseudo_label(pseudo_pred, pseudo_box_label,unsup_batch,save_visual=False)#形状为(batchsize，2)，每个值为[box，mask]
                            #回调伪标签的水平翻转
                            pseudo_label = self.horizonFlip(pseudo_label,horizons)
                            #对原始的unsup_batch进行强增强（旋转，裁剪，水平翻转，擦除），返回变化矩阵调整伪标签
                            unsup_s = deepcopy(unsup_batch)
                            #把伪标签和无监督数据对齐
                            unsup_s = align_batch(unsup_s,pseudo_label)
                            
                            #增强无监督数据和伪标签
                            unsup_s = self.strong_aug(unsup_s)
                            # if RANK in {-1,0} and (epoch % 5 == 0):
                            #     visual_augment(unsup_s,"teacher_vis",epoch)  #教师模型伪标签可视化
                            
                            
                        unlabel_pred, _ = self.model(unsup_s)
                        self.unsup_loss, unsup_loss_items = self.model.module.new_unsup_loss(unlabel_pred, unsup_s)#教师产生伪标签，学生学习
                        #step3：学生模型预测伪标签监督教师
                        with torch.no_grad():
                            self.teacher_model.train()
                            self.model.eval()
                            pseudo_pred1,_ = self.model(unsup_w)#学生模型生成伪标签供教师学习
                            # pseudo_pred1[0] = [p.detach() for p in pseudo_pred1[0]]
                            # pseudo_pred1[1] = [p.detach() for p in pseudo_pred1[1]]
                            pseudo_box_label1 = self.get_pseudo_box_label(pseudo_pred1[0])
                            pseudo_label1 = self.get_pseudo_label(pseudo_pred1, pseudo_box_label1,unsup_w,save_visual=False)
                            pseudo_label1 = self.horizonFlip(pseudo_label1,horizons)
                            unsup_s1 = deepcopy(unsup_batch)#后续同步两种增强
                            unsup_s1 = align_batch(unsup_s1,pseudo_label1)
                            unsup_s1 = self.strong_aug(unsup_s1)
                            # if RANK in {-1,0} and (epoch % 5 == 0):
                            #     visual_augment(unsup_s,"student_vis",epoch)  #学生模型伪标签可视化
                        unlabel_pred1, _ = self.teacher_model(unsup_s1)
                        self.unsup_loss1, unsup_loss_items1 = self.teacher_model.module.new_unsup_loss(unlabel_pred1,unsup_s1)#学生产生伪标签，教师学习

      
                    #self.unsup_loss = self.model.unsup_loss(unlabel_pred, pseudo_label)
                        #self.unsup_loss = self.model.module.unsup_loss(unlabel_pred, pseudo_label)
                    
                    #self.unsup_loss = self.cal_unsup_loss(unlabel_pred, pseudo_label)#需要伪标签的batch[batch_idx], batch[cls],batch[bboxes],batch[masks]
                        if  self.auto_train:
                            self.total_loss = self.unsup_loss
                            self.total_loss1 = self.unsup_loss1

                            self.loss_items = unsup_loss_items
                            self.loss_items1 = unsup_loss_items1

                            self.loss = self.unsup_loss
                            self.loss1 = self.unsup_loss1
                        else:
                            self.total_loss = ((1-lamda) * self.loss + (lamda*1) * self.unsup_loss)
                            self.total_loss1 = ((1-lamda) * self.loss1 + (lamda*1) * self.unsup_loss1)
                    self.teacher_model.train()
                    self.model.train()
                    #整合loss
                    # if RANK != 1:
                        # self.loss *= world_size
                        # self.total_loss *= world_size
                        # self.total_loss1 *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    self.tloss1 = (
                        (self.tloss1 * i + self.loss_items1) / (i + 1) if self.tloss1 is not None else self.loss_items1
                    )

                    #反向传播
                self.scaler.scale(self.total_loss).backward()
                self.scaler.scale(self.total_loss1).backward()
                    
                #self.scaler.scale(self.loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.optimizer_teacher_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break
                
                # if lamda != 0:
                #     self.update_teacher_model()

                # Log
                # if RANK in {-1, 0}:
                #     loss_length = self.tloss.shape[0] + (self.tloss1.shape[0]) if len(self.tloss.shape) else 1
                #     pbar.set_description(
                #     ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                #     % (
                #           f"{epoch + 1}/{self.epochs}",
                #           f"{self._get_memory():.3g}G", # (GB) GPU memory util
                #           *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                #           batch["cls"].shape[0], # batch size, i.e. 8
                #           batch["img"].shape[-1], # imgsz, i.e 640
                #           *(self.tloss1 if loss_length > 1 else torch.unsqueeze(self.tloss, 0)), # losses
                #       )
                #     )
                #     self.run_callbacks("on_batch_end")
                #     if self.args.plots and ni in self.plot_idx:
                #     self.plot_training_samples(batch, ni)
                        
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] + (self.tloss1.shape[0]) if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            unsup_batch["img"].shape[0] if self.auto_train else batch["cls"].shape[0],  # batch size, i.e. 8
                            unsup_batch["img"].shape[-1] if self.auto_train else batch["img"].shape[-1],  # imgsz, i.e 640
                            *(self.tloss1 if loss_length > 1 else torch.unsqueeze(self.tloss, 0)), # losses
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        if self.auto_train:
                            self.plot_training_samples(unsup_batch, ni)
                        else:
                            self.plot_training_samples(batch, ni)


                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                # self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                # self.teacher_ema.update_attr(self.teacher_model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop or not self.auto_train:
                    self.metrics, self.fitness = self.validate()
                    self.metrics1,self.fitness1 = self.teacher_validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

                # #如果目前的结果就是最好结果，额外保存一份学生模型用来监督教师模型的学习
                # if self.stopper.best_epoch == (epoch + 1):
                #     self.best_student = deepcopy(self.model)
                #     print(f"保存新的最佳学生，最佳epoch为：{epoch + 1}")
                #     self.reset_teacher = True
                # self.update_count += 1

                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch = epoch + 1 

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def _setup_train(self, world_size):
        
        """Builds dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()

        self.model = self.model.to(self.device)

        if self.teacher_model is not None:
            teacher_ckpt = self.setup_teacher_model()
            self.teacher_model = self.teacher_model.to(self.device)
            

        self.best_student = None
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        if self.teacher_model is not None:
            for k, v in self.teacher_model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
                if any(x in k for x in freeze_layer_names):
                    LOGGER.info(f"Freezing layer '{k}'")
                    v.requires_grad = False
                elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                    LOGGER.info(
                        f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                        "See ultralytics.engine.trainer for customization of frozen layers."
                    )
                    v.requires_grad = True
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            # if self.teacher_model is not None:
            #     self.amp_unsup = torch.tensor(check_amp(self.teacher_model),device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
            # if self.amp_unsup is not None:
            #     dist.broadcast(self.unsup_amp, src=0) 
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True,broadcast_buffers=False)
            if self.teacher_model is not None:
                self.teacher_model = nn.parallel.DistributedDataParallel(self.teacher_model, device_ids=[RANK], find_unused_parameters=True,broadcast_buffers=False)

        self.first_unsup = True

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        self.unsup_loader = self.get_dataloader(self.unsupset, batch_size=batch_size, rank=LOCAL_RANK, mode="unsup_train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            self.teacher_validator = self.get_teacher_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.teacher_model is not None:
                self.teacher_ema = ModelEMA(self.teacher_model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self.teacher_optimizer = self.build_optimizer(
            model=self.teacher_model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        if teacher_ckpt is not None:
            self.resume_unsup_training(teacher_ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def setup_teacher_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.teacher_model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.teacher_model, None
        ckpt = None
        if str(self.teacher_model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.teacher_model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.teacher_model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if self.teacher_model:
            self.teacher_model.nc = self.data["nc"]
            self.teacher_model.names = self.data["names"]
            self.teacher_model.args = self.args
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_dataset(self):
        #获取半监督数据集格式
        self.data, self.unsup_data = check_semi_seg_dataset(self.args.data, self.args.unsup_data)
        return self.data["train"], self.unsup_data["train"], self.data["val"] or self.data["test"]

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        if self.teacher_optimizer is not None:
            self.teacher_scheduler = optim.lr_scheduler.LambdaLR(self.teacher_optimizer, lr_lambda=self.lf)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        imgs = batch["img"]
        batch['is_label'] = True
        if self.args.multi_scale:
            
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch
        
    def preprocess_semi_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["is_label"] = False
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_pseudo_box_label(self,preds):
        conf_thres = 0.25 
        iou_thres = 0.45
        classes = None
        max_wh = 4096#一个足够大的值作为偏移，防止不同类别的框相互抑制
        nc = 1

        bs = preds.shape[0] # batch size
        nc = nc or (preds.shape[1] - 4) # number of classes
        nm = preds.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        xc = preds[:, 4:mi].amax(1) > conf_thres  # candidates

        #setting
        multi_label = False

        preds = preds.transpose(-1, -2)
        tmp = preds
        preds[..., :4] = xywh2xyxy(tmp[..., :4]) #原地修改表示格式，从xywh变成xyxy

        output = [torch.zeros((0, 6 + nm), device=preds.device)] * bs
        max_nms = 30000

        for xi, x in enumerate(preds):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            c = x[:, 5:6] * max_wh  # classes
            scores = x[:, 4]

            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS（最大抑制)
            
            output[xi] = x[i]

        return output
    


    def get_pseudo_label(self, preds, p, batch, save_visual):#代码还有bug要改
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        pseudo_label = []
        img_shape = batch['img'][0][0].shape
        
        for i,pred in enumerate(p):
            if len(pred) == 0:          
                masks=None
            else:           
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img_shape, upsample=True)
                #pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], ori_img_size)

                # if save_visual is True:#可视化伪标签
                #     results = []
                #     file_path = batch['im_file'][i]
                #     im0 = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)
                #     # if not isinstance(im0, list):  # input images are a torch.Tensor, not a list
                #     #     im0 = ops.convert_torch2numpy_batch(im0)
                #     results.append(Results(im0, path=file_path, names=self.model.module.names, boxes=pred[:,:6], masks=masks))
                #     print("start to save")
                #     for item in results:
                #         path = item.path
                #         path = path.split('/')[-1]
                #         print(path)
                #         item.save(filename=os.path.join("runs", "train_vis", path))

            pseudo_label.append([pred[:, :6],masks])

        return pseudo_label
    
    def augment_batch(self,batch,aug_prob):

        import albumentations as A

        aug_batch = batch
        imgs = aug_batch["img"]
        return aug_batch

    def resume_unsup_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.teacher_optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.teacher_ema and ckpt.get("ema"):
            self.teacher_ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.teacher_emaema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.teacher_model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def optimizer_teacher_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.teacher_optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.teacher_optimizer)
        self.scaler.update()
        self.teacher_optimizer.zero_grad()
        if self.teacher_ema:
            self.teacher_ema.update(self.teacher_model)

    def teacher_validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.teacher_validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness
    
    def get_teacher_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks, teacher_model=self.teacher_model
        )

def check_semi_seg_dataset(dataset, unsup_dataset):
    file = check_file(dataset)
    unsup_file = check_file(unsup_dataset)

    data = yaml_load(file, append_filename=True)
    unsup_data = yaml_load(unsup_file, append_filename=True)

    # Resolve paths
    extract_dir = ""
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    unsup_path = Path(extract_dir or unsup_data.get("path") or Path(unsup_data.get("yaml_file", "")).parent)

        # Set paths
    data["path"] = path  # download scripts
    unsup_data["path"] = unsup_path
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    k = "train"
    if unsup_data.get(k):
        if isinstance(unsup_data[k], str):
            y = (path / unsup_data[k]).resolve()
            if not y.exists() and unsup_data[k].startswith("../"):
                y = (path / unsup_data[k][3:]).resolve()
            unsup_data[k] = str(y)
        else:
            unsup_data[k] = [str((path / y).resolve()) for y in unsup_data[k]]
    
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    return data, unsup_data

def combine_batch(label_batch, unlabel_batch):
    def cat(key):
        return torch.cat([label_batch[key], unlabel_batch[key]], dim=0)
    
    batch = {
        key: cat(key) for key in label_batch.keys() if key in unlabel_batch
    }

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def visualize_pseudo_label(img, boxes, masks=None, alpha=0.5, save_path=None):
    """
    img: 原图 (H, W, 3)，numpy 格式，uint8
    boxes: Tensor [N, 6]，xyxy + conf + cls
    masks: Tensor [N, H, W]，已经 resize 成图像大小
    """
    img = img.copy()
    img = np.ascontiguousarray(img)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    # Draw masks
    if masks is not None:
        for m in masks:
            colored_mask = np.zeros_like(img)
            color = np.random.randint(0, 255, size=3)
            for c in range(3):
                colored_mask[..., c] = (m > 0.5) * color[c]

            img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)

    # Draw boxes
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"{int(cls)}:{conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255,255,255), 1, lineType=cv2.LINE_AA)

    plt.imshow(img)
    if save_path:
        plt.savefig(save_path)
    plt.axis("off")
    plt.clf()

def align_batch(batch,labels):

    #将伪标签和batch中的masks，cls，bboxes，batch_idx对上
    batch_idx = []
    device = batch['img'].device
    batch['cls'] = batch['cls'].to(device)
    batch['bboxes'] = batch['bboxes'].to(device)
    batch['batch_idx'] = batch['batch_idx'].to(device)
    batch["conf"] = torch.empty(0)
    batch["conf"] = batch["conf"].to(device)
    masks = torch.empty(0,640,640)
    masks = masks.to(device)
    for i in range(len(labels)):
        if labels[i][0].shape[0] == 0:
            continue
        else:
            for j in range(labels[i][0].shape[0]):
                batch_idx.append(i)
               
                batch["cls"] = torch.cat((batch['cls'],torch.zeros(1,device=device)))
                batch['bboxes'] = torch.cat((batch['bboxes'],labels[i][0][j][:4].unsqueeze(0).to(device)))
                masks = torch.cat((masks,labels[i][1][j].unsqueeze(0)))
                batch["conf"] = torch.cat((batch["conf"],labels[i][0][j][4].unsqueeze(0).to(device)))
    batch_idx = torch.tensor(batch_idx,device=device)
    batch['batch_idx'] = batch_idx
    batch['masks'] = masks
    batch["cls"] = batch['cls'].unsqueeze(-1)
    return batch

def visual_augment(batch,aug_mode,epoch):
    #需要的输入内容：
    # image：原始图像，H,W,3 npArray
    # masks: n,(H,W) npArray
    # bboxes: n,4 xmin,ymin,xmax,ymax npArray
    # labels: 实例的标签类型
    # class_name_map: 类型名称映射
    # mask_alpha: 掩码透明度 (0-1)
    # bbox_color: 边界框颜色 (BGR)
    # bbox_thickness: 边界框线宽
    images = batch['img']
    masks = batch['masks']
    bboxes = batch['bboxes']
    labels = batch['cls']
    class_name_map = None
    mask_alpha = 1
    bbox_color = (0, 255, 0)
    bbox_thickness = 1

    batch_idx = batch['batch_idx'].tolist()
    for i in range(batch["img"].shape[0]):
        indices = [j for j,x in enumerate(batch_idx) if x == i]
        image = images[i]
        mask = masks[indices]
        boxes = bboxes[indices]
        label = labels[indices]

        image = image.permute(1,2,0).cpu().numpy()
        img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
        mask = mask.cpu().numpy()
        box = boxes[:,:4].cpu().numpy()
        vis = draw_instances(img,mask,box,label,class_name_map,mask_alpha,bbox_color,bbox_thickness)
        plt.imshow(vis)
        plt.axis("off")

        output_path = Path("runs") / "aug_vis" / aug_mode / f"{epoch}_{i}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.clf()


# def _do_train(self, world_size=1):
        
    #     if world_size > 1:
    #         self._setup_ddp(world_size)
    #     self._setup_train(world_size)

    #     nb = len(self.train_loader) # 每个 epoch 的 batch 数
    #     nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup 的总 iteration 数
    #     last_opt_step = -1  # warmup 的总 iteration 数
    #     self.epoch_time = None
    #     self.epoch_time_start = time.time()
    #     self.train_time_start = time.time()
    #     self.run_callbacks("on_train_start")
    #     LOGGER.info(
    #         f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
    #         f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
    #         f"Logging results to {colorstr('bold', self.save_dir)}\n"
    #         f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    #     )
    #     #训练后期关闭 Mosaic 数据增强，并标记这几个 batch 做可视化。
    #     if self.args.close_mosaic:
    #         base_idx = (self.epochs - self.args.close_mosaic) * nb
    #         self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

    #     epoch = self.start_epoch
    #     self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
    #     while True:
    #         self.epoch = epoch
    #         self.run_callbacks("on_train_epoch_start")
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
    #             self.scheduler.step()

    #         self.model.train()
    #         #self.teacher_model.eval()
    #         if RANK != -1:#打乱一次乱序数据的同步
    #             self.train_loader.sampler.set_epoch(epoch)
    #             self.unsup_loader.sampler.set_epoch(epoch)
    #         #pbar = enumerate(self.train_loader)
    #         pbar = enumerate(zip(self.train_loader, self.unsup_loader))
    #         # Update dataloader attributes (optional)
    #         if epoch == (self.epochs - self.args.close_mosaic):
    #             self._close_dataloader_mosaic()
    #             self.train_loader.reset()
    #             self.unsup_loader.reset()
    #         if RANK in {-1, 0}:
    #             LOGGER.info(self.progress_string())
    #             pbar = TQDM(enumerate(zip(self.train_loader, self.unsup_loader)), total=nb)
    #         self.tloss = None
        
    #         if epoch < 150:
    #             lamda = 0
    #         else:
    #             lamda = min(((0.5*epoch + (0.05 * self.epochs)) / self.epochs),0.4)
                
    #         for i, (batch,unsup_batch) in pbar:

    #             self.run_callbacks("on_train_batch_start")
    #             #self.semi_epoch = 250

    #             # Warmup
    #             ni = i + nb * epoch
    #             if ni <= nw:
    #                 xi = [0, nw]  # x interp
    #                 self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
    #                 for j, x in enumerate(self.optimizer.param_groups):
    #                     # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
    #                     x["lr"] = np.interp(
    #                         ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
    #                     )
    #                     if "momentum" in x:
    #                         x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])


    #             with autocast(self.amp):

    #                 batch = self.preprocess_batch(batch)
    #                 unsup_batch = self.preprocess_semi_batch(unsup_batch)

    #                 #step1：学生模型根据标签数据进行学习
    #                 self.loss, self.loss_items = self.model(batch)
    #                 self.total_loss = self.loss

    #                 if lamda != 0:
    #                     #第一次预测时同步教师模型和学生模型
    #                     if self.first_unsup:
    #                         self.first_unsup = False
    #                         with torch.no_grad():
    #                             if self.best_student is not None:
    #                                 self.teacher_model = deepcopy(self.best_student)
    #                             else:
    #                                 self.teacher_model = deepcopy(self.model)
    #                             for p in self.teacher_model.parameters():
    #                                 p.requires_grad = False
    #                             self.teacher_model.eval()
    #                     #unsup_batch完全未经过数据增强
    #                     #复制一份deepcopy的unsup_batch，并对unsup_batch进行弱增强（水平翻转，高斯噪声）
    #                     unsup_w = deepcopy(unsup_batch)
    #                     unsup_w,horizons = self.weak_aug(unsup_w)#进行增强并记录哪些图像被翻转了
    #                     unsup_w2,horizons2 = self.weak_aug(unsup_w)
    #                     #教师模型预测弱增强结果
    #                     with torch.no_grad():

    #                     #step2：教师模型预测结果监督学生
    #                         pseudo_pred, _ = self.teacher_model(unsup_w)
    #                         #将预测结果转换为伪标签
    #                         pseudo_box_label = self.get_pseudo_box_label(pseudo_pred[0])
    #                         pseudo_label = self.get_pseudo_label(pseudo_pred, pseudo_box_label,unsup_batch,save_visual=False)#形状为(batchsize，2)，每个值为[box，mask]
    #                         #回调伪标签的水平翻转
    #                         pseudo_label = self.horizonFlip(pseudo_label,horizons)
    #                         #对原始的unsup_batch进行强增强（旋转，裁剪，水平翻转，擦除），返回变化矩阵调整伪标签
    #                         unsup_s = deepcopy(unsup_batch)
    #                         #把伪标签和无监督数据对齐
    #                         unsup_s = align_batch(unsup_s,pseudo_label)
    #                         # if RANK in {-1,0} and (epoch % 5 == 0):
    #                         #     visual_augment(unsup_s,"weak_aug",epoch)  #增强前可视化
    #                         #增强无监督数据和伪标签
    #                         unsup_s = self.strong_aug(unsup_s)
    #                         # if RANK in {-1,0} and (epoch % 5 == 0):
    #                         #     visual_augment(unsup_s,"strong_aug",epoch)  #增强后的可视化
                            
    #                     #step3：学生模型预测伪标签监督教师
    #                         pseudo_pred2,_ = self.model.module(unsup_w2)
    #                         pseudo_box_label2 = self.get_pseudo_box_label(pseudo_pred2[0])
    #                         pseudo_label2 = self.get_pseudo_label(pseudo_pred2, pseudo_box_label2,unsup_batch,save_visual=False)#形状为(batchsize，2)，每个值为[box，mask]
    #                         pseudo_label2 = self.horizonFlip(pseudo_label2,horizons2)
    #                         unsup_s2 = deepcopy(unsup_batch)
    #                         unsup_s2 = align_batch(unsup_s2,pseudo_label)
    #                         unsup_s2 = self.strong_aug(unsup_s2)

    #                     #根据增强数据进行学生模型预测
    #                     unlabel_pred, _ = self.model(unsup_s)
    #                     unlabel_pred2, _ = self.teacher_model(unsup_s2)

    #                     self.unsup_loss,_ = self.model.module.new_unsup_loss(unlabel_pred, unsup_s)
    #                     self.unsup_loss2,_ = self.teacher_model.module.new_unsup_loss(unlabel_pred2, unsup_s2)
    #                 #self.unsup_loss = self.model.unsup_loss(unlabel_pred, pseudo_label)
    #                     #self.unsup_loss = self.model.module.unsup_loss(unlabel_pred, pseudo_label)
                    
    #                 #self.unsup_loss = self.cal_unsup_loss(unlabel_pred, pseudo_label)#需要伪标签的batch[batch_idx], batch[cls],batch[bboxes],batch[masks]
    #                     self.total_loss = ((1-lamda) * self.loss + (lamda*0.5) * self.unsup_loss + (lamda*0.5) * self.unsup_loss2)

    #                 #整合loss
    #                 if RANK != 1:
    #                     self.loss *= world_size
    #                     self.total_loss *= world_size
    #                 self.tloss = (
    #                     (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
    #                 )

    #                 #反向传播
    #             self.scaler.scale(self.total_loss).backward()
                    
    #             #self.scaler.scale(self.loss).backward()

    #             if ni - last_opt_step >= self.accumulate:
    #                 self.optimizer_step()
    #                 last_opt_step = ni

    #                 # Timed stopping
    #                 if self.args.time:
    #                     self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
    #                     if RANK != -1:  # if DDP training
    #                         broadcast_list = [self.stop if RANK == 0 else None]
    #                         dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
    #                         self.stop = broadcast_list[0]
    #                     if self.stop:  # training time exceeded
    #                         break
                
    #             if lamda != 0:
    #                 self.update_teacher_model()

    #             # Log
    #             if RANK in {-1, 0}:
    #                 loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
    #                 pbar.set_description(
    #                     ("%11s" * 2 + "%11.4g" * (2 + loss_length))
    #                     % (
    #                         f"{epoch + 1}/{self.epochs}",
    #                         f"{self._get_memory():.3g}G",  # (GB) GPU memory util
    #                         *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
    #                         batch["cls"].shape[0],  # batch size, i.e. 8
    #                         batch["img"].shape[-1],  # imgsz, i.e 640
    #                     )
    #                 )
    #                 self.run_callbacks("on_batch_end")
    #                 if self.args.plots and ni in self.plot_idx:
    #                     self.plot_training_samples(batch, ni)

    #             self.run_callbacks("on_train_batch_end")

    #         self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
    #         self.run_callbacks("on_train_epoch_end")
    #         if RANK in {-1, 0}:
    #             final_epoch = epoch + 1 >= self.epochs
    #             self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

    #             # Validation
    #             if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
    #                 self.metrics, self.fitness = self.validate()
    #             self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
    #             self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch

    #             #如果目前的结果就是最好结果，额外保存一份学生模型用来监督教师模型的学习
    #             if self.stopper.best_epoch == (epoch + 1):
    #                 self.best_student = deepcopy(self.model)
    #                 print(f"保存新的最佳学生，最佳epoch为：{epoch + 1}")
    #                 self.reset_teacher = True
    #             self.update_count += 1

    #             if self.args.time:
    #                 self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

    #                 # Save model
    #             if self.args.save or final_epoch:
    #                 self.save_model()
    #                 self.run_callbacks("on_model_save")

    #         # Scheduler
    #         t = time.time()
    #         self.epoch_time = t - self.epoch_time_start
    #         self.epoch_time_start = t
    #         if self.args.time:
    #             mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
    #             self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
    #             self._setup_scheduler()
    #             self.scheduler.last_epoch = self.epoch  # do not move
    #             self.stop |= epoch >= self.epochs  # stop if exceeded epochs
    #         self.run_callbacks("on_fit_epoch_end")
    #         self._clear_memory()

    #         # Early Stopping
    #         if RANK != -1:  # if DDP training
    #             broadcast_list = [self.stop if RANK == 0 else None]
    #             dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
    #             self.stop = broadcast_list[0]
    #         if self.stop:
    #             break  # must break all DDP ranks
    #         epoch = epoch + 1 

    #     if RANK in {-1, 0}:
    #         # Do final val with best.pt
    #         seconds = time.time() - self.train_time_start
    #         LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
    #         self.final_eval()
    #         if self.args.plots:
    #             self.plot_metrics()
    #         self.run_callbacks("on_train_end")
    #     self._clear_memory()
    #     self.run_callbacks("teardown")