import os
from typing import List, Optional, Sequence, Tuple, Union
from numpy import random
import mmcv
import numpy as np
import copy
import json
from pycocotools.coco import COCO
from tqdm import tqdm


def find_inside_bboxes(bboxes, img_h, img_w):
    """Find bboxes as long as a part of bboxes is inside the image.

    Args:
        bboxes (Tensor): Shape (N, 4).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        Tensor: Index of the remaining bboxes.
    """
    inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) \
        & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
    return inside_inds


class DataAugmentation:
    def __init__(self,
                 image_root,
                 json_file,
                 num_rounds=8,
                 num_copies=2,
                 img_scale: Tuple[int, int] = (448, 448),
                 center_ratio_range: Tuple[float, float] = (1.0, 1.0),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0) -> None:
        assert isinstance(img_scale, tuple)
        self.image_root = image_root
        self.json_file = json_file
        self.data = COCO(json_file)
        image_ids = list(self.data.imgs.keys())
        self.image_start_id = max(image_ids) + 1
        ann_ids = list(self.data.anns.keys())
        self.annotation_start_id = max(ann_ids) + 1
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.num_copies = num_copies
        self.num_rounds = num_rounds

        os.makedirs(os.path.join(image_root, "mosaic"), exist_ok=True)


    def copy(self):
        add_imgs = dict()
        add_img2anns = dict()

        for img_id, anns in tqdm(self.data.imgToAnns.items()):
            img_info = copy.deepcopy(self.data.imgs[img_id])
            img_info.update(id=self.image_start_id)
            add_imgs[self.image_start_id] = img_info

            add_anns = []
            for ann in anns:
                add_ann = copy.deepcopy(ann)
                add_ann.update(image_id=self.image_start_id,
                               id=self.annotation_start_id)
                add_anns.append(add_ann)
                self.annotation_start_id += 1

            add_img2anns[self.image_start_id] = add_anns
            self.image_start_id += 1
        add_anns_list = []
        for add_anns in add_img2anns.values():
            add_anns_list += add_anns
        return list(add_imgs.values()), add_anns_list

    def mosaic(self):
        image_ids = list(self.data.imgs.keys())
        num_mosaics = len(image_ids) // 4
        random.shuffle(image_ids)

        add_anns = []
        add_images = []

        for i in tqdm(range(num_mosaics)):
            results = self.coco2mmdet(image_ids[i * 4])
            mix_results = [self.coco2mmdet(image_ids[i * 4 + 1]),
                           self.coco2mmdet(image_ids[i * 4 + 2]),
                           self.coco2mmdet(image_ids[i * 4 + 3])]
            results.update(mix_results=mix_results)

            results = self.transform(results)

            img = results['img']
            height, width = img.shape[:2]
            add_images.append(dict(id=self.image_start_id,
                                   file_name=f"mosaic/{self.image_start_id}.jpg",
                                   height=height, width=width
                                   ))
            mmcv.imwrite(img, os.path.join(self.image_root, f"mosaic/{self.image_start_id}.jpg"))

            gt_bboxes = results['gt_bboxes']
            gt_bboxes_labels = results['gt_bboxes_labels']
            gt_bboxes[:, 2:] = gt_bboxes[:, 2:] - gt_bboxes[:, :2]


            for gt_bbox, gt_bbox_label in zip(gt_bboxes, gt_bboxes_labels):
                x, y, w, h = gt_bbox.tolist()
                add_anns.append(dict(id=self.annotation_start_id,
                                     bbox=[x,y,w,h],
                                     category_id=int(gt_bbox_label),
                                     image_id=self.image_start_id,
                                     area=w * h,
                                     iscrowd=0))
                self.annotation_start_id += 1

            self.image_start_id += 1

        return add_images, add_anns


    def offline_aug(self):
        images = list(self.data.imgs.values())
        annotations = list(self.data.anns.values())

        for _ in range(self.num_copies):
            copied_images, copied_anns = self.copy()
            images += copied_images
            annotations += copied_anns

        for _ in range(self.num_rounds):
            mosaic_images, mosaic_anns = self.mosaic()
            images += mosaic_images
            annotations += mosaic_anns

        with open(self.json_file.replace('.json', '_mosaic.json'), "w") as f:
            json.dump(dict(images=images,
                           annotations=annotations, categories=list(self.data.cats.values())), f)

    def coco2mmdet(self, image_id):
        anns = self.data.imgToAnns[image_id]
        img_info = self.data.imgs[image_id]
        image_path = os.path.join(self.image_root, img_info['file_name'])

        img = mmcv.imread(image_path)
        results = dict(img=img)
        gt_bboxes = []
        gt_bboxes_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']

            gt_bboxes.append([x, y, x + w, y + h])
            gt_bboxes_labels.append(category_id)

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_bboxes_labels = np.array(gt_bboxes_labels)

        results.update(gt_bboxes=gt_bboxes,
                       gt_bboxes_labels=gt_bboxes_labels)
        return results

    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']

            padw = x1_p - x1_c
            padh = y1_p - y1_c

            if gt_bboxes_i.shape[0] > 0:
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)

        mosaic_bboxes =np.concatenate(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)

        if self.bbox_clip_border:
            mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                             2 * self.img_scale[1])
            mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                             2 * self.img_scale[0])

        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])

        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]

        results['img'] = mosaic_img
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        return results

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord


if __name__ == "__main__":
    data_aug = DataAugmentation(image_root="datasets/ovd360/data_pre_contest/data_pre_contest/train",
                                json_file="datasets/ovd360/train_eng.json")
    data_aug.offline_aug()