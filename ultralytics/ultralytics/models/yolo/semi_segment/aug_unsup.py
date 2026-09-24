import albumentations as A
import numpy as np
import torch
from collections import defaultdict
import cv2

class StrongNoiseBlurAug:
    def __init__(self):
        self.transform = A.ReplayCompose([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),  
            A.BBoxSafeRandomCrop(erosion_rate=0.0),
            A.CoarseDropout(min_holes=3,
                            max_holes=6,
                            min_height=10,
                            max_height=20,
                            min_width=10,
                            max_width=20,
                            fill_value=0,  # or "random_uniform"
                            mask_fill_value=None,
                            p=0.8),
            #A.GaussNoise(p=0.5), 
            #A.Rotate(limit=30, p=0.8,fit_output=False), 
            A.Resize(height=640,width=640),
        ],bbox_params=A.BboxParams(format='pascal_voc',min_visibility=0.6,label_fields=['cls','pos_indice','label_idx'],filter_invalid_bboxes=True))
        # ,
        # keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])

    def __call__(self, batch):
        """
        参数 batch: Dict，包含 "img"、"img_shape"、其他字段。
        可兼容多数 YOLO/半监督任务数据结构。
        """
        images = batch["img"]  # shape: (B, C, H, W)
        bboxes = batch["bboxes"]
        masks = batch["masks"]
        box_cls = batch["cls"]
        batch_idx = batch["batch_idx"].tolist()
        device = images.device
        images_augs = []
        replays = []
        for i in range(batch["img"].shape[0]):
            image_aug = defaultdict(lambda:{
                "img":[],
                "bboxes":[],
                "masks":[],
                "cls":[],
                "idx":[]
            })
            image_aug["img"] = images[i]
            indice = [j for j,x in enumerate(batch_idx) if x == i]
            image_aug["boxes"] = bboxes[indice]
            image_aug["masks"] = masks[indice]
            image_aug["cls"] = box_cls[indice]
            image_aug['idx'] = []
        
            images_augs.append(image_aug)
        idx_count = 0
        for item in images_augs:
            img_np = item["img"].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC, 转为 numpy
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np
            mask_np = item["masks"].cpu().numpy()
            masks_np = np.array([mask for mask in mask_np])
            bboxes_np = item["boxes"].cpu().numpy()
            boxes_np = np.array([box for box in bboxes_np])
            cls_np = item['cls'].cpu().numpy()
            clses_np = np.array([cls for cls in cls_np])
            if masks_np.size != 0:
                label_idx = batch_idx[idx_count:idx_count+masks_np.shape[0]]
                idx_count += masks_np.shape[0]
                label_idx = np.array(label_idx)
                pos_indices = np.array(range(masks_np.shape[0]))
                transformed = self.transform(image=img_np, masks=masks_np,bboxes=boxes_np,cls=clses_np,pos_indice=pos_indices,label_idx=label_idx)
                item["boxes"] = torch.tensor(transformed["bboxes"],device=device)
                all_masks = torch.tensor(transformed["masks"],device=device)
                indices = transformed["pos_indice"]
                item["masks"] = all_masks[indices]
                item['cls'] = torch.tensor(transformed['cls'],device=device).view(-1,1)
                item['idx'] = torch.tensor(transformed['label_idx'],device=device)
            else:
                transformed = self.transform(image=img_np,bboxes=np.empty((0,4)),cls=np.empty(0),pos_indice=np.empty(0),label_idx=np.empty(0))
            aug_img = transformed["image"]
            aug_img = torch.from_numpy(aug_img).float().permute(2, 0, 1) / 255.0
            item["img"] = aug_img
            #replays.append(transformed["replay"])
        
        new_images = []
        new_bboxes = torch.empty(0,4)
        new_bboxes = new_bboxes.to(device)
        new_masks = torch.empty(0,640,640)
        new_masks = new_masks.to(device)
        new_cls = torch.empty(0,1)
        new_cls = new_cls.to(device)
        new_idx = torch.empty(0)
        new_idx = new_idx.to(device)

        for item in images_augs:
            new_images.append(item["img"])
            new_bboxes = torch.cat([new_bboxes,item["boxes"]])
            new_masks = torch.cat([new_masks,item["masks"]])
            new_cls = torch.cat([new_cls,item['cls']])
            if not isinstance(item['idx'],list):
                new_idx = torch.cat([new_idx,item['idx']])
        batch["img"] = torch.stack(new_images).to(device)
        batch["bboxes"] = new_bboxes
        batch["masks"] = new_masks
        batch['cls'] = new_cls
        batch['batch_idx'] = new_idx
        
        return batch
        #return batch,replays

class WeekAugmentation:
    def __init__(self):
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.2),
            A.Resize(height=640,width=640),
        ],save_key="replay")
        
    def __call__(self,batch):
        images = batch["img"]
        images_aug = []
        is_horizons = []

        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC, 转为 numpy
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np
            aug_dict = self.transform(image=img_np)
            aug = aug_dict["image"]
            replay = aug_dict["replay"]
            is_horizon = check_horizon(replay)
            aug = torch.from_numpy(aug).float().permute(2, 0, 1) / 255.0  # HWC -> CHW，归一化
            images_aug.append(aug)
            is_horizons.append(is_horizon)

        batch["img"] = torch.stack(images_aug).to(images.device)
        batch["aug_type"] = is_horizons
        return batch,is_horizons
    
def check_horizon(replay):
    transforms = replay.get('transforms', {})
    for t in transforms:
        if t.get('__class_fullname__') == "HorizontalFlip":
            return t.get('applied', False)
    return False  # 或 False

class HorizonFlip:
    def __init__(self):
        self.transform = A.Compose([A.HorizontalFlip(p=1.0),A.Resize(height=640,width=640)],
                                   A.BboxParams(format="pascal_voc",
                                                min_height=0,
                                                min_width=0,
                                                min_visibility=0,
                                                filter_invalid_bboxes=True,
                                                label_fields=['pos_indices']
                                                ),
                                    )
    def __call__(self,pseudo_labels,horizons):
        device = pseudo_labels[0][0].device
        lost_count = 0
        for i in range(len(pseudo_labels)):
            if horizons[i] and pseudo_labels[i][1] is not None:
                start_len = pseudo_labels[i][1].shape[1]
                masks = pseudo_labels[i][1].cpu().numpy()
                mask_np = np.array([mask for mask in masks])
                pos_indices = np.array(range(mask_np.shape[0]))
                boxes = pseudo_labels[i][0][:,:4].cpu().numpy()
                boxes_np = np.array([box for box in boxes])
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                transformed = self.transform(image=dummy_image, masks=mask_np, bboxes=boxes_np,pos_indices=pos_indices)
                all_masks = torch.tensor(transformed["masks"],device=device)
                indices = transformed["pos_indices"]
                pseudo_labels[i][1] = all_masks[indices]
                pseudo_labels[i][0] = pseudo_labels[i][0][indices]
                pseudo_labels[i][0][:,:4] = torch.tensor(transformed["bboxes"],device=device)
                end_len = pseudo_labels[i][1].shape[1]

                if(start_len != end_len):
                    lost_count += (start_len-end_len)
                    print(f"lost {lost_count} in the horizon")
        return pseudo_labels
        