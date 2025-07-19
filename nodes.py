import os
import torch
import node_helpers
import numpy as np

from PIL import Image, ImageOps, ImageSequence

class LoadImageFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {
                    "tooltip": "Absolute or relative path to the image file.",
                    "multiline": False,
                }),
                "RGBA": ("BOOLEAN", {
                    "tooltip": "Controls whether to include the Alpha channel in the 'image' output.  The 'mask' output will always include this data.",
                    "default": False,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "FLOAT")
    RETURN_NAMES = ("images", "mask", "frame_count", "fps")
    OUTPUT_TOOLTIPS = (
        "The image data, either as a single image or a set of frames.",
        "Any transparency data found in the Alpha channel or transparent palette index.",
        "Total number of frames loaded from the image.",
        "Frames-per-second, as reported from the image metadata.  This may be zero, if the data could not be found, or the image doesn't animate."
    )
    FUNCTION = "load_image_from_path"
    DESCRIPTION = "Load an image from a filename path.  Supports frame data, in a similar fashion to the 'Load Image' node."

    CATEGORY = "image"

    # Mostly a fork of "Load Images" from base, with some modifications
    def load_image_from_path(self, image_path, RGBA=False):
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        frames = 0
        duration = fps = 0.00
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            image = i
            if not RGBA:
                image = image.convert('RGB')

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            frames += 1
            if 'duration' in i.info:
                duration += i.info['duration']

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if frames > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        if frames > 1 and duration > 0:
            fps = frames / duration * 1000

        return (output_image, output_mask, frames, float(fps))

NODE_CLASS_MAPPINGS = {
    "LoadAnimAdv_LoadImageFromPath": LoadImageFromPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAnimAdv_LoadImageFromPath": "Load Image From Path"
}
