import os
import torch
import node_helpers
import numpy as np
import mimetypes

from PIL import Image, ImageOps, ImageSequence

from comfy_api.input_impl import VideoFromFile

### Standalone functions

# Shamelessly stolen from comfyui-inspire-pack
sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)"
]

def sort_by(items, base_path='.', method=None):
    def fullpath(x): return os.path.join(base_path, x)

    def get_timestamp(path):
        try:
            return os.path.getmtime(path)
        except FileNotFoundError:
            return float('-inf')

    if method == "Alphabetical (ASC)":
        return sorted(items)
    elif method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    elif method == "Numerical (ASC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    else:
        return items

def list_files_from_directory(directory: str, file_list_cap: int, file_start_index: int, include_subfolders: bool, valid_extensions: str, sort_method: str):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory} cannot be found.'")

    valid_extensions_list = list(map(
        lambda e: '.' + e.strip(),
        valid_extensions.strip().split(',')
    ))

    # Scan for files
    dir_files = []
    if include_subfolders:
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions_list):
                    path = os.path.join(root, file)
                    dir_files.append(path)
    else:
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in valid_extensions_list):
                path = os.path.join(directory, file)
                dir_files.append(path)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")

    dir_files = sort_by(dir_files, '.', sort_method)

    # start at start_index
    dir_files = dir_files[file_start_index:]

    filenames = []
    for path in dir_files:
        if os.path.isdir(path) and os.path.ex:
            continue
        if file_list_cap > 0 and len(filenames) >= file_list_cap:
            break
        filenames.append(path)

    return (filenames,)

def load_image_video_from_path(path: str, RGBA: bool=False):
    # Figure out if it's an image or video file
    mime_type = mimetypes.guess_type(path)[0]
    file_type = None
    if mime_type == None:
        # Resort to our own file extension matching
        if path.split('.')[-1].lower() in ['mp4','api','mov','mkv']:
            file_type = 'video'
    else:
        file_type = mime_type.split('/', maxsplit=1)[0]

    img, audio = None, None
    frames = 0
    duration = fps = 0.00

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    if file_type == 'video':
        components = VideoFromFile(path).get_components()
        img, audio, fps = components.images, components.audio, float(components.frame_rate)
        output_image = img

        # TODO: Support video tranparency, though not too many formats themselves support it
        for i in img:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_masks.append(mask.unsqueeze(0))
        output_mask = torch.cat(output_masks, dim=0)
    else:
        img = node_helpers.pillow(Image.open, path)
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

        if frames > 1 and duration > 0 and fps == 0:
            fps = frames / duration * 1000

    return (output_image, output_mask, audio, frames, float(fps))

### Node definitions

class LoadImageVideoFromPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {
                    "tooltip": "Absolute or relative path to the image/video file.",
                    "placeholder": "input/images/image.png",
                    "multiline": False,
                }),
                "RGBA": ("BOOLEAN", {
                    "tooltip": "Controls whether to include the Alpha channel in the 'image' output.  The 'mask' output will always include this data.",
                    "default": False,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("images", "mask", "audio", "frame_count", "fps")
    OUTPUT_TOOLTIPS = (
        "The image data, either as a single image or a set of frames.",
        "Any transparency data found in the Alpha channel or transparent palette index.",
        "The audio from the video, if any.",
        "Total number of frames loaded from the image.",
        "Frames-per-second, as reported from the image metadata.  This may be zero, if the data could not be found, or the image doesn't animate."
    )
    FUNCTION = "execute"
    DESCRIPTION = "Load an image or video from a filename path.  Supports animation frame data, audio, and video."

    CATEGORY = "image"

    def execute(self, path: str, RGBA: bool):
        return load_image_video_from_path(path=path, RGBA=RGBA)

class ListFilesFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {
                    "tooltip": "Absolute or relative path to the image directory.",
                    "multiline": False,
                }),
            },
            "optional": {
                "file_list_cap": ("INT", {
                    "tooltip": "Maximum amount of files to list out.  Zero means no maximum.",
                    "default": 0,
                    "min":     0,
                    "step":    1,
                }),
                "file_start_index": ("INT", {
                    "tooltip": "Index (zero-based) to start listing files, based on the collected filenames from the directory search.",
                    "default": 0,
                    "min":     0,
                    "step":    1,
                }),
                "include_subfolders": ("BOOLEAN", {
                    "tooltip": "Controls whether to recursively search for files within subdirectories.",
                    "default": False,
                }),
                "valid_extensions": ("STRING", {
                    "tooltip": "Comma-delimited list of file extensions to search for.",
                    "default": "jpg,jpeg,png,webp,gif,tga",
                }),
                "sort_method": (sort_methods, {
                    "tooltip": "Controls how to sort the image filename list.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filenames",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "The list of filenames that match the inputs specified.  Input nodes that do not support lists of files will be executed one-at-a-time.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Lists files from a directory."

    CATEGORY = "util"

    def execute(self, directory: str, file_list_cap: int, file_start_index: int, include_subfolders: bool, valid_extensions: str, sort_method: str):
        return list_files_from_directory(
            directory=directory,
            file_list_cap=file_list_cap,
            file_start_index=file_start_index,
            include_subfolders=include_subfolders,
            valid_extensions=valid_extensions,
            sort_method=sort_method,
        )

NODE_CLASS_MAPPINGS = {
    "LoadAnimAdv_LoadImageVideoFromPath": LoadImageVideoFromPath,
    "LoadAnimAdv_ListFilesFromDirectory": ListFilesFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAnimAdv_LoadImageVideoFromPath": "Load Image/Video From Path",
    "LoadAnimAdv_ListFilesFromDirectory": "List Files From Directory",
}
