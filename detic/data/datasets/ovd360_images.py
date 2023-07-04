# Copyright (c) Facebook, Inc. and its affiliates.
import os

from .imagenet import custom_register_imagenet_instances
from .ovd360 import _get_builtin_metadata


_CUSTOM_SPLITS_IMAGENET = {
    "ovd365_images": ("ovd360/crawled_images/", "ovd360/crawled_images.json"),
    "ovd365_images_large": ("ovd360/crawled_images_large/", "ovd360/crawled_images_large.json"),
}

for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET.items():
    custom_register_imagenet_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file),
        os.path.join("datasets", image_root),
    )

