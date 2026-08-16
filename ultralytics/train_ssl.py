# from ultralytics import YOLO

# #model = YOLO("/data1/zhaohaowei/peak_detection/yolo/ultralytics/yolo11m-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/data1/zhaohaowei/peak_detection/yolo/ultralytics/runs/segment/seg-with-pre-train/weights/best.pt") #加载训练好的全监督训练集
# #model = YOLO("yolo11m-seg.yaml") #从头训练模型

# # Train the model with 2 GPUs
# results = model.train(task="semi_segment",data="dataset.yaml",unsup_data="unsup_dataset.yaml",batch=128, epochs=900, imgsz=640, device=[0,1,2,3])
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
sys.path.append(os.path.dirname(__file__))

from ultralytics.models.yolo.semi_segment import SemiSegmentationTrainer  # 替换为你的路径

# 在一切导入之前设置环境变量
overrides = dict(
    task="semi_segment", 
    model="/data1/zhaohaowei/peak_detection/yolo/ultralytics/yolo11m_pre_new.pt",
    #unsup_model = "/data1/zhaohaowei/peak_detection/yolo/ultralytics/yolo11m-seg.pt",
    unsup_model = "/data1/zhaohaowei/peak_detection/yolo/ultralytics/runs/semi_segment/train13/weights/best.pt",
    data="dataset.yaml",
    unsup_data="unsup_dataset.yaml",
    imgsz=640,
    batch=32,
    epochs=500,
    patience=200,
    device="0,1,2,3",
    optimizer = "Adam",
    self_train = True
    #augment=True
)#调整无监督数据集batch比例在train的412行get_dataloader输入的参数里改，默认1:2
trainer = SemiSegmentationTrainer(overrides=overrides)
trainer.train()
