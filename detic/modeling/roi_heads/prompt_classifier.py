# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import json
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec
from detic.modeling import clip as CLIP


class PromptClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,         # json file here
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        clip_cfg = None
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.prompt_length = clip_cfg.PROMPT_LEN

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)

        # Load category names
        with open(zs_weight_path, 'r') as f:
            categories = json.load(f)['categories']
        cat_names = [x['name'] for x in
                     sorted(categories, key=lambda x: x['id'])]
        tokens = CLIP.tokenize_dynamic(cat_names, truncate=True)
        self.register_buffer('tokens', tokens)
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_classes, prompt_length, clip_cfg.WORD_DIM))

        self.clip, self.clip_preprocess = CLIP.load(name=clip_cfg.NAME,
                                                    use_image_encoder=False,
                                                    download_root=None)
        self.clip.init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            "clip_cfg": cfg.MODEL.CLIP,
        }

    @property
    def class_embeddings(self):
        cls = self.clip.encode_text_with_prompt(self.tokens, prompt=self.prompt_embeddings, normalize=True)
        bg = torch.zeros_like(cls[:1])
        return torch.cat([cls, bg], dim=0)


    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        assert classifier is None
        zs_weight = self.class_embeddings.T
        x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
