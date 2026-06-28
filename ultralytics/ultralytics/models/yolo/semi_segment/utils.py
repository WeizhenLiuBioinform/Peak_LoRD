import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
import random

def draw_instances(image, masks, bboxes, labels, class_name_map=None, 
                  mask_alpha=1, bbox_color=(0, 255, 0), bbox_thickness=2):
    """
    在图像上绘制实例分割结果：掩码+边界框+标签
    
    参数:
        image: 原始图像 (H, W, 3)
        masks: 掩码列表 [ (H, W) 的二进制数组, ... ]
        bboxes: 边界框列表 [[x_min, y_min, x_max, y_max], ...]
        labels: 每个实例的类别标签
        class_name_map: 类别ID到名称的映射字典
        mask_alpha: 掩码透明度 (0-1)
        bbox_color: 边界框颜色 (BGR)
        bbox_thickness: 边界框线宽
    """
    # 创建基础图像
    vis_image = image.copy()
    
    # 1. 绘制掩码（每个实例使用随机颜色）
    for mask in masks:
        if mask is None or mask.size == 0:
            continue
            
        # 为每个实例生成随机颜色
        color = [255,0,0]
        
        # 创建彩色掩码覆盖层
        mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        mask_overlay[mask > 0] = color
        
        # 将掩码混合到图像上
        vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, mask_alpha, 0)
    
    # 2. 绘制边界框和标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for bbox, label in zip(bboxes, labels):
        # 转换边界框坐标
        try:
            if len(bbox) < 4:
                continue
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
        except (ValueError, TypeError):
            continue
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), bbox_color, bbox_thickness)
        
        # 准备标签文本
        label_name = str(label) if class_name_map is None else class_name_map.get(label, str(label))
        
        # 计算文本位置
        (text_width, text_height), baseline = cv2.getTextSize(
            label_name, font, font_scale, font_thickness
        )
        
        # 绘制文本背景
        cv2.rectangle(
            vis_image,
            (x_min, y_min - text_height - baseline),
            (x_min + text_width, y_min),
            bbox_color,
            -1  # 填充矩形
        )
        
        # 绘制标签文本
        cv2.putText(
            vis_image,
            label_name,
            (x_min, y_min - baseline),
            font,
            font_scale,
            (255, 255, 255),  # 白色文本
            font_thickness
        )
    
    return vis_image


def visualize_instance_segmentation(dataset, idx=0, samples=3):
    """
    可视化实例分割数据增强效果
    
    参数:
        dataset: 包含图像、掩码、边界框和标签的数据集
        idx: 要可视化的数据集索引
        samples: 要生成的增强样本数量
    """
    # 创建可视化专用transform（移除Normalize/ToTensor）
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        # 保留原始的bbox_params和mask_params
        bbox_params = getattr(dataset.transform, 'bbox_params', None)
        mask_params = getattr(dataset.transform, 'mask_params', None)
        
        vis_transform = A.Compose(
            vis_transform_list,
            bbox_params=bbox_params,
            mask_params=mask_params
        )
    else:
        print("Warning: Using original transform without stripping Normalize/ToTensor")
        vis_transform = dataset.transform
    
    # 创建画布
    fig, ax = plt.subplots(samples + 1, 2, figsize=(12, 4 * (samples + 1)))
    fig.suptitle('Instance Segmentation Augmentation Visualization', fontsize=16)
    
    # 获取原始数据
    original_transform = dataset.transform
    dataset.transform = None
    data = dataset[idx]
    dataset.transform = original_transform
    
    # 解包数据（根据数据集结构调整）
    image = data['image'] if isinstance(data, dict) else data[0]
    masks = data['masks'] if 'masks' in data else []
    bboxes = data['bboxes'] if 'bboxes' in data else []
    labels = data['labels'] if 'labels' in data else []
    
    # 可视化原始数据
    original_vis = draw_instances(image, masks, bboxes, labels)
    
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis('off')
    
    ax[0, 1].imshow(original_vis)
    ax[0, 1].set_title("Original Instances")
    ax[0, 1].axis('off')
    
    # 可视化增强样本
    for i in range(samples):
        try:
            # 应用增强
            augmented = vis_transform(
                image=image,
                masks=masks,
                bboxes=bboxes,
                labels=labels
            ) if vis_transform else {
                'image': image,
                'masks': masks,
                'bboxes': bboxes,
                'labels': labels
            }
            
            # 获取增强结果
            aug_image = augmented['image']
            aug_masks = augmented['masks']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['labels']
            
            # 绘制实例可视化
            aug_vis = draw_instances(aug_image, aug_masks, aug_bboxes, aug_labels)
            
            # 显示结果
            ax[i+1, 0].imshow(aug_image)
            ax[i+1, 0].set_title(f"Augmented Image {i+1}")
            ax[i+1, 0].axis('off')
            
            ax[i+1, 1].imshow(aug_vis)
            ax[i+1, 1].set_title(f"Augmented Instances {i+1}")
            ax[i+1, 1].axis('off')
            
        except Exception as e:
            print(f"Error in augmentation {i+1}: {str(e)}")
            # 出错时显示原始图像
            ax[i+1, 0].imshow(image)
            ax[i+1, 0].set_title(f"Augmentation Error {i+1}")
            ax[i+1, 0].axis('off')
            
            ax[i+1, 1].imshow(original_vis)
            ax[i+1, 1].set_title(f"Augmentation Error {i+1}")
            ax[i+1, 1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


