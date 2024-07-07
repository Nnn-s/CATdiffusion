import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import patchify
import os
from typing import Union

import PIL.Image
import PIL.ImageOps
import requests
import random

def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def crop_512(image, mask, random_flag=True):
        image_np = np.array(image)
        mask_np = np.array(mask)
        h, w = mask_np.shape

        x = (mask_np / 255.).round() > 0.5
        indices = np.where(x)
        if len(indices[0]) == 0 or len(indices[1]) == 0:
            top = random.randint(0, h - 512)
            left = random.randint(0, w - 512)
        else:
            a = np.min(indices[0])
            b = np.max(indices[0])
            c = np.min(indices[1])
            d = np.max(indices[1])

            if random_flag:
                low_top = min(max(0, b - 512), a)
                high_top = max(min(a, h - 512), 0)
                top = 0 if h == 512 else random.randint(low_top, high_top)

                low_left = min(max(0, d - 512), c)
                high_left = max(min(c, w - 512), 0)
                left = 0 if w == 224 else random.randint(low_left, high_left)
            else:
                abm = (a + b) // 2
                cdm = (c + d) // 2
                top = min(max(0, abm - 256), h // 2 - 256)
                left = min(max(0, cdm - 256), w // 2 - 256)
                top = 0 if h == 512 else top
                left = 0 if w == 512 else left

        cropped_image_np = image_np[top:top+512, left:left+512]
        cropped_mask_np = mask_np[top:top+512, left:left+512]

        cropped_image = Image.fromarray(cropped_image_np.astype(np.uint8)).convert("RGB")
        cropped_mask = Image.fromarray(cropped_mask_np.astype(np.uint8)).convert("L")
        return cropped_image, cropped_mask

def AddOriginalImageSizeAsTupleAndCropToSquare(img, original_h, original_w):
    original_size_as_tuple = torch.tensor([original_h, original_w])
    # x['jpg'] should be chw tensor  in [-1, 1] at this point
    size = min(img.shape[1], img.shape[2])
    delta_h = img.shape[1] - size
    delta_w = img.shape[2] - size
    assert not all(
        [delta_h, delta_w]
    )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
    top = np.random.randint(0, delta_h + 1)
    left = np.random.randint(0, delta_w + 1)
    img = transforms.functional.crop(
        img, top=top, left=left, height=size, width=size
    )
    crop_coords_top_left = torch.tensor([top, left])
    return img, original_size_as_tuple, crop_coords_top_left

def patchify_mask(mask, num_patches=16):
    h, w = mask.shape
    assert w % num_patches == 0 and h == w
    step = patch_h = patch_w = w // num_patches
    patched_masks = patchify.patchify(mask, (patch_h, patch_w), step) # n_rows, n_colos, 1, h, w, numpy array
    patched_masks = (patched_masks.reshape((num_patches, num_patches, patch_h * patch_w)) / 255.).round()
    patched_masks = np.sum(patched_masks, axis=-1) > 0
    return patched_masks.reshape(num_patches * num_patches)

def masking(image, mask,return_pil = True):
    image_np = np. array(image)
    mask_np = np.array(mask)[:, :, None] / 255.
    
    
    if return_pil:
        masked_image_np = image_np * (1 - mask_np) + np.ones_like(image_np) * mask_np * 127 # for clip
        masked_image = Image.fromarray(masked_image_np.astype(np.uint8))
        return masked_image
    else:
        masked_image_np = image_np * (1 - mask_np) + np.zeros_like(image_np) * mask_np  # for sd-inpaint
        return masked_image_np

def preprocess_mask(mask,aug=False):
    if not isinstance(mask, torch.Tensor):
        # preprocess mask
        if isinstance(mask, Image.Image) or isinstance(mask, np.ndarray):
            mask = [mask]

        if isinstance(mask, list):
            if isinstance(mask[0], Image.Image):
                mask = [np.array(m.convert("L")).astype(np.float32) / 255.0 for m in mask]
            if isinstance(mask[0], np.ndarray):
                mask = np.stack(mask, axis=0) if mask[0].ndim < 3 else np.concatenate(mask, axis=0)
                mask = torch.from_numpy(mask)
            elif isinstance(mask[0], torch.Tensor):
                mask = torch.stack(mask, dim=0) if mask[0].ndim < 3 else torch.cat(mask, dim=0)

    # Batch and add channel dim for single mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)




    # Check mask is in [0, 1]
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("`mask_image` should be in [0, 1] range")
    # Binarize mask
    if aug:
        mask[mask > 0] =1
    else:
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

    return mask

def generate_max_image(box, h, w):
    max_image = np.zeros((h, w), dtype=np.uint8)
    xmin, ymin, xmax, ymax = box
    max_image = max_image.astype(np.float32)  # Convert to float32
    max_image[int(ymin * h):int(ymax * h), int(np.ceil(xmin * w)):int(np.ceil(xmax * w))] = 1
    return max_image

def generate_box_from_mask(mask):
    non_zero_rows, non_zero_cols = np.nonzero(mask)
    if len(non_zero_rows) == 0:
        return None  

    left = np.min(non_zero_cols)
    top = np.min(non_zero_rows)
    right = np.max(non_zero_cols)
    bottom = np.max(non_zero_rows)



    height = mask.shape[0]
    width = mask.shape[1]

    if (top-bottom)*(left-right) > 0.65*height*width:
        return None
    #if box is too large,use the origin mask

    box = [left / width, top / height, (right + 1) / width, (bottom + 1) / height]  


    return box


def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):


    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], Image.Image):
            mask = [i.resize((width, height), resample=Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image



