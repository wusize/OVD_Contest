# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import os
import cv2
from tqdm import tqdm
from glob import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default=r'G:\BaiduNetdiskDownload\crawled_images')
    parser.add_argument('--json_path', default=r"G:\BaiduNetdiskDownload\json_pre_contest\json_pre_contest\test_eng.json")
    parser.add_argument('--out_path', default=r"G:\BaiduNetdiskDownload\json_pre_contest\json_pre_contest\crawled_images.json")
    args = parser.parse_args()

    print('Loading LVIS meta')
    data = json.load(open(args.json_path, 'r'))
    print('Done')

    categories = data['categories']
    count = 0
    images = []
    image_counts = {}

    for cat in tqdm(categories):
        cat_id = cat['id']
        cat_image_path = os.path.join(args.image_path, str(cat_id))
        image_files = glob(os.path.join(cat_image_path, "*.jpg"))

        for image_file in image_files:
            basename = os.path.basename(image_file)
            count = count + 1
            img = cv2.imread(image_file)
            if img is None:
                continue
            h, w = img.shape[:2]
            image = {
                'id': count,
                'file_name': f"{cat_id}/{basename}",
                'pos_category_ids': [cat_id],
                'width': w,
                'height': h
            }
            images.append(image)

    out = {'categories': categories, 'images': images, 'annotations': []}
    print('Writing to', args.out_path)
    json.dump(out, open(args.out_path, 'w'))
