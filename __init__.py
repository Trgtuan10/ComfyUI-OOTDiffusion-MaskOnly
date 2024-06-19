import os
import warnings
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location

_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}

_category_readable = {
    "Upper body": "upperbody",
    "Lower body": "lowerbody",
    "Dress": "dress",
}

class OOTDMasking:
    display_name = "OOTDiffusion Masking"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cloth_image": ("IMAGE",),
                "model_image": ("IMAGE",),
                # Openpose from comfyui-controlnet-aux not work
                # "keypoints": ("POSE_KEYPOINT",),
                "category": (list(_category_readable.keys()),),
            }
        }

    RETURN_TYPES = ("IMAGE")
    RETURN_NAMES = ("image_masked")
    FUNCTION = "Masking"

    CATEGORY = "Masking"

    def generate(
        self, pipe: OOTDiffusion, model_image, category
    ):
        # if model_image.shape != (1, 1024, 768, 3) or (
        #     cloth_image.shape != (1, 1024, 768, 3)
        # ):
        #     raise ValueError(
        #         f"Input image must be size (1, 1024, 768, 3). "
        #         f"Got model_image {model_image.shape} cloth_image {cloth_image.shape}"
        #     )
        category = _category_readable[category]

        # (1,H,W,3) -> (3,H,W)
        model_image = model_image.squeeze(0)
        model_image = model_image.permute((2, 0, 1))
        model_image = to_pil_image(model_image)
        if model_image.size != (768, 1024):
            print(f"Inconsistent model_image size {model_image.size} != (768, 1024)")
        model_image = model_image.resize((768, 1024))

        model_parse, _ = pipe.parsing_model(model_image.resize((384, 512)))
        keypoints = pipe.openpose_model(model_image.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            pipe.model_type,
            _category_get_mask_input[category],
            model_parse,
            keypoints,
            width=384,
            height=512,
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_image, mask)

        masked_vton_img = masked_vton_img.convert("RGB")
        masked_vton_img = to_tensor(masked_vton_img)
        masked_vton_img = masked_vton_img.permute((1, 2, 0)).unsqueeze(0)

        return  masked_vton_img


_export_classes = [
    OOTDMasking,
]

NODE_CLASS_MAPPINGS = {c.__name__: c for c in _export_classes}

NODE_DISPLAY_NAME_MAPPINGS = {
    c.__name__: getattr(c, "display_name", c.__name__) for c in _export_classes
}
