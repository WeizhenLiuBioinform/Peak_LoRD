# Semi-supervised detection training script (semi_detect task)
# Adapted from train_ssl.py for detection-only mode (no mask branch)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
sys.path.append(os.path.dirname(__file__))

from ultralytics.models.yolo.semi_detect import SemiDetectionTrainer

# ============================================================
# TODO: 根据实际情况修改以下路径和参数
# ============================================================

overrides = dict(
    task="semi_detect",

    # 学生模型：使用检测模型配置（非分割）
    # 可从 yaml 从头训练，或加载预训练 .pt 权重
    model="yolov8n.yaml",                    # 检测模型配置，或替换为你的 .pt 路径

    # 教师模型（可选）：预训练权重路径，设为 None 则从学生模型复制初始化
    # unsup_model=None,
    unsup_model=None,

    # 有标签数据集配置
    data="dataset.yaml",

    # 无标签数据集配置
    unsup_data="unsup_dataset.yaml",

    # 训练参数
    imgsz=640,
    batch=32,
    epochs=500,
    patience=200,
    device="0,1,2,3",
    optimizer="Adam",

    # 自训练模式：
    #   False = 标准半监督（有标签 + 无标签联合训练）
    #   True  = 纯自训练（仅无标签数据，lambda=1.0）
    self_train=False,

    # 其他可选参数：
    # lr0=0.0005,            # 初始学习率
    # lrf=0.01,              # 最终学习率比例
    # weight_decay=0.0005,   # 权重衰减
    # warmup_epochs=3.0,     # 预热轮数
    # close_mosaic=10,       # 最后 N 个 epoch 关闭 mosaic
    # amp=True,              # 混合精度训练
    # pretrained=True,       # 使用预训练权重（bool）或指定权重路径（str）
)

trainer = SemiDetectionTrainer(overrides=overrides)
trainer.train()
