import os
import torch
import numpy as np
import mimetypes
import collections.abc

import nodes
import node_helpers
import folder_paths

from PIL import Image, ImageOps, ImageSequence, ImageStat
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types.node_typing import IO
from operator import itemgetter

### Useful globals

sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)"
]
aspect_ratio_resize_methods = [
    "crop",
    "pad",
    "stretch",
]
image_loader_common_inputs = {
    "RGBA": ("BOOLEAN", {
        "tooltip": "Controls whether to include the Alpha channel in the 'image' output.  The 'mask' output will always include this data.",
        "default": False,
    }),
    "width": ("INT", {
        "tooltip": "Final width of images, after resizing.  Zero means to keep the original width.",
        "default": 0,
        "min":     0,
        "step":    16,
    }),
    "height": ("INT", {
        "tooltip": "Final height of images, after resizing.  Zero means to keep the original height.",
        "default": 0,
        "min":     0,
        "step":    16,
    }),
    "keep_aspect_ratio": (aspect_ratio_resize_methods, {
        "tooltip": "What action to take, when the image doesn't match the expected aspect ratio.",
    }),
}

### Standalone functions

# Shamelessly stolen from comfyui-inspire-pack
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

# Shamelessly stolen from comfyui-kjnodes
def resize_with_aspect_ratio(img, width, height, mode):
    if mode == "stretch":
        return img.resize((width, height), Image.Resampling.LANCZOS)

    img_width, img_height = img.size
    aspect_ratio = img_width / img_height
    target_ratio = width / height

    if mode == "crop":
        # Calculate dimensions for center crop
        if aspect_ratio > target_ratio:
            # Image is wider - crop width
            new_width = int(height * aspect_ratio)
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
            left = (new_width - width) // 2
            return img.crop((left, 0, left + width, height))
        else:
            # Image is taller - crop height
            new_height = int(width / aspect_ratio)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
            top = (new_height - height) // 2
            return img.crop((0, top, width, top + height))

    elif mode == "pad":
        pad_color = get_edge_color(img)
        # Calculate dimensions for padding
        if aspect_ratio > target_ratio:
            # Image is wider - pad height
            new_height = int(width / aspect_ratio)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
            padding = (height - new_height) // 2
            padded = Image.new('RGBA', (width, height), pad_color)
            padded.paste(img, (0, padding))
            return padded
        else:
            # Image is taller - pad width
            new_width = int(height * aspect_ratio)
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
            padding = (width - new_width) // 2
            padded = Image.new('RGBA', (width, height), pad_color)
            padded.paste(img, (padding, 0))
            return padded

def get_edge_color(img):
    """Sample edges and return dominant color"""
    width, height = img.size
    img = img.convert('RGBA')

    # Create 1-pixel high/wide images from edges
    top = img.crop((0, 0, width, 1))
    bottom = img.crop((0, height-1, width, height))
    left = img.crop((0, 0, 1, height))
    right = img.crop((width-1, 0, width, height))

    # Combine edges into single image
    edges = Image.new('RGBA', (width*2 + height*2, 1))
    edges.paste(top, (0, 0))
    edges.paste(bottom, (width, 0))
    edges.paste(left.resize((height, 1)), (width*2, 0))
    edges.paste(right.resize((height, 1)), (width*2 + height, 0))

    # Get median color
    stat = ImageStat.Stat(edges)
    median = tuple(map(int, stat.median))
    return median

### Functions used by nodes

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

def load_image_video_from_path(path: str, RGBA: bool=False, width: int=0, height: int=0, keep_aspect_ratio: str="crop"):
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
    w,  h  = None, None
    fw, fh = None, None

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

            if len(output_images) == 0:
                w,  h  = i.size[0], i.size[1]
                fw, fh = width, height
                if fw == 0: fw = w
                if fh == 0: fh = h

            # Each frame of animation should match the previous frame's dimensions
            if i.size[0] != w or i.size[1] != h:
                continue

            # Capture frame data
            frames += 1
            if 'duration' in i.info:
                duration += i.info['duration']

            # Resize image to maximum dimensions
            if fw != w or fh != h:
                i = resize_with_aspect_ratio(i, fw, fh, keep_aspect_ratio)

            # Convert to RGB, if desired
            image = i
            if not RGBA:
                image = image.convert('RGB')

            # NumPY/Torch tensor conversion
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

            # Add to output lists
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

def select_indexes_from_any(obj, indexes_to_select: str):
    len_shape = None
    if isinstance(obj, torch.Tensor):
        len_shape = obj.shape[0]
    elif isinstance(obj, list):
        len_shape = len(obj)
    else:
        raise TypeError(f"Object is a '{type(obj).__qualname__}', not a listable type.")

    selected_index: list[int] = []
    idxs_shape: list[int] = list( range(0, len_shape) )
    for s in indexes_to_select.strip().split(','):
        if ':' in s:
            # https://docs.python.org/3/library/stdtypes.html#range
            ranges = s.strip().split(':', maxsplit=2)
            ranges = [int(r.strip()) if r.strip()!='' else None for r in ranges]

            selected_index.extend( idxs_shape[slice(*ranges)] )
        else:
            i = int(s.strip())
            selected_index.append(i)

    if len(selected_index) == 0:
        raise ValueError(f"No indexes selected for '{indexes_to_select}'.")

    if isinstance(obj, torch.Tensor):
        return obj[selected_index]
    elif isinstance(obj, list):
        lst = []
        if len(selected_index) == 1:
            # itemgetter returns the object itself, not a tuple, with a single index
            lst.append(obj[selected_index[0]])
        else:
            lst.append(itemgetter(*selected_index)(obj))
        return lst
    else:
        raise TypeError(f"Object is a '{type(obj).__qualname__}', not a listable type.")

### Node definitions

class LoadImageVideo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image", "video"])
        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True
                }),
                **image_loader_common_inputs,
            }
        }

    RETURN_TYPES = ("IMAGE",  "MASK", "AUDIO", "INT",         "FLOAT")
    RETURN_NAMES = ("images", "mask", "audio", "frame_count", "fps")
    OUTPUT_TOOLTIPS = (
        "The image data, either as a single image or a set of frames.",
        "Any transparency data found in the Alpha channel or transparent palette index.",
        "The audio from the video, if any.",
        "Total number of frames loaded from the image.",
        "Frames-per-second, as reported from the image metadata.  This may be zero, if the data could not be found, or the image doesn't animate."
    )
    FUNCTION = "execute"
    DESCRIPTION = "Load an image or video.  Supports animation frame data, audio, and video."

    CATEGORY = "LoadAnimAdv/image"

    def execute(self, image: str, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        return load_image_video_from_path(
            path=image_path,
            **kwargs,
        )

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
                **image_loader_common_inputs,
            }
        }

    RETURN_TYPES = ("IMAGE",  "MASK", "AUDIO", "INT",         "FLOAT")
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

    CATEGORY = "LoadAnimAdv/image"

    def execute(self, path: str, **kwargs):
        return load_image_video_from_path(
            path=path,
            **kwargs,
        )

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
                    "control_after_generate": True,
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

    CATEGORY = "LoadAnimAdv/util"

    def execute(self, directory: str, file_list_cap: int, file_start_index: int, include_subfolders: bool, valid_extensions: str, sort_method: str):
        return list_files_from_directory(
            directory=directory,
            file_list_cap=file_list_cap,
            file_start_index=file_start_index,
            include_subfolders=include_subfolders,
            valid_extensions=valid_extensions,
            sort_method=sort_method,
        )

class LoadImagesVideosFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {
                    "tooltip": "Absolute or relative path to the image directory.",
                    "multiline": False,
                }),
                **image_loader_common_inputs,
                "image_file_load_cap": ("INT", {
                    "tooltip": "Maximum amount of image files (not frames) to load.  Zero means no maximum.",
                    "default": 0,
                    "min":     0,
                    "step":    1,
                }),
                "image_file_start_index": ("INT", {
                    "tooltip": "Index (zero-based) to start loading files, based on the collected filenames from the directory search.",
                    "default": 0,
                    "min":     0,
                    "step":    1,
                }),
                "frame_indexes_to_select": ("STRING", {
                    "tooltip": (
                        "Frame indexes (zero-based) to select from each image file.  " +
                        "Supports ':' for range selection (including ':#', '#:' formats), '::#' & '#::#' & '#:#:#' formats for step selection, " +
                        "',' for multiple entries, and negative numbers for selecting from the end.  Note that audio is not sliced based on frame selection, " +
                        "which may cause audio desyncs.  (This may be fixed in the future.)"
                    ),
                    "default": "0:",
                }),
                "total_frame_load_cap": ("INT", {
                    "tooltip": "Maximum amount of total frames to load.  Zero means no maximum.",
                    "default": 0,
                    "min":     0,
                    "step":    1,
                }),
                "flatten_frames": ("BOOLEAN", {
                    "tooltip": (
                        "Controls whether to flatten the frames from multiple files into a single image set.  " +
                        "This impacts all outputs except 'image_path'."
                    ),
                    "default": False,
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
                    "tooltip": "Controls how to sort the image filename list."
                }),
            },
        }

    RETURN_TYPES    = ("IMAGE", "MASK", "AUDIO", "INT",          "FLOAT", "STRING")
    RETURN_NAMES    = ("image", "mask", "audio", "frame_counts", "fps",   "image_path")
    OUTPUT_IS_LIST  = (True,    True,   True,    True,           True,    True)
    OUTPUT_TOOLTIPS = (
        "The image data, either as a single image or a set of frames.",
        "Any transparency data found in the Alpha channel or transparent palette index.",
        "The audio from the video, if any.",
        "Total number of frames loaded from all images.",
        "Frames-per-second, as reported from all of the image metadata.  This may be zero, if the data could not be found, or the image doesn't animate.  " +
            "If flattened, multiple animations with different FPSs will be averaged and may have unpredictable results.",
        "Filenames of the loaded images.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Load a series of images from directory path.  Supports loading frame data."

    CATEGORY = "LoadAnimAdv/image"

    def execute(
        self, directory: str,
        # from image_loader_common_inputs
        RGBA: bool, width: int, height: int, keep_aspect_ratio: str,

        image_file_load_cap: int = 0, image_file_start_index: int = 0,
        frame_indexes_to_select: str = '0:', total_frame_load_cap: int = 0, flatten_frames: bool = False,
        include_subfolders=False, valid_extensions: str = 'jpg,jpeg,png,webp,gif,tga', sort_method=None
    ):
        filenames = list_files_from_directory(
            directory=directory,
            file_list_cap=image_file_load_cap,
            file_start_index=image_file_start_index,
            include_subfolders=include_subfolders,
            valid_extensions=valid_extensions,
            sort_method=sort_method,
        )[0]

        (images, masks, audios, frame_counts, fps_list, image_paths) = ([], [], [], [], [], [])
        total_frame_count = 0

        # File loading loop
        for filename in filenames:
            # Load the image/video
            frames, mask, audio, frame_count, fps = load_image_video_from_path(
                path=filename,
                RGBA=RGBA,
                width=width,
                height=height,
                keep_aspect_ratio=keep_aspect_ratio,
            )

            # Select frames from the image
            new_frames = select_indexes_from_any(obj=frames, indexes_to_select=frame_indexes_to_select)
            new_mask   = select_indexes_from_any(obj=mask,   indexes_to_select=frame_indexes_to_select)

            new_frame_count = new_frames.size(dim=0)
            if total_frame_load_cap > 0 and total_frame_count + new_frame_count > total_frame_load_cap:
                frame_diff = total_frame_load_cap - total_frame_count
                new_frames = new_frames[:frame_diff]
                new_mask   = new_mask  [:frame_diff]

            # Add to lists
            if flatten_frames and len(images) > 0:
                images[0] = torch.cat((images[0], new_frames), dim=0)
                masks [0] = torch.cat((masks [0], new_mask  ), dim=0)
                audios[0] = torch.cat((audios[0], audio     ), dim=0)

                frame_counts[0] += new_frame_count
                fps_list[0]     += fps  # averaged later
            else:
                images      .append(new_frames)
                masks       .append(new_mask)
                audios      .append(audio)
                frame_counts.append(frame_count)
                fps_list    .append(fps)

            image_paths.append(filename)

            total_frame_count += new_frame_count
            if total_frame_load_cap > 0 and total_frame_count >= total_frame_load_cap:
                break

        if flatten_frames:
            fps_list[0] = fps_list[0] / len(image_paths)

        return (images, masks, audios, frame_counts, fps_list, image_paths,)

class SelectIndexesFromImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image set to select from.",
                }),
                "indexes_to_select": ("STRING", {
                    "tooltip":
                        "Indexes (zero-based) to select.  " +
                        "Supports ':' for range selection (including ':#', '#:' formats), '::#' & '#::#' & '#:#:#' formats for step selection, " +
                        "',' for multiple entries, and negative numbers for selecting from the end."
                    ,
                    "default": "0:",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = (
        "The selected image set.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Select specific images from a set, using a variety of supported syntaxes."

    CATEGORY = "LoadAnimAdv/image"

    def execute(self, images, indexes_to_select):
        # Single image:             type=Tensor size=torch.Size([1, 1000, 1000, 3])  # or ', 4' for RGBA
        # Normal animations:        type=Tensor size=torch.Size([75, 1024, 1024, 3])
        # Multiple item animations: (runs it multiple times)
        # Mutliple items with INPUT_IS_LIST=True:
        #     images:    type=list size=2
        #     images[0]: type=Tensor size=torch.Size([75, 1024, 1024, 3])
        #     images[1]: type=Tensor size=torch.Size([75, 1024, 1024, 3])

        if isinstance(images, torch.Tensor):
            new_images = select_indexes_from_any(obj=images, indexes_to_select=indexes_to_select)
            return (new_images,)
        elif isinstance(images, list):
            # XXX: This is essentially INPUT_IS_LIST support, but we turned it off, in favor of letting ComfyUI
            # run the node multiple times.
            if len(images) == 0:
                return (images,)
            elif isinstance(images[0], torch.Tensor):
                new_list = []
                for i in images:
                    new_list.append( select_indexes_from_any(obj=i, indexes_to_select=indexes_to_select) )
                return (new_list,)
        else:
            raise TypeError(f"'images' is not a listable type.")

class SelectIndexesFromAny:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (IO.ANY, {
                    "tooltip": "Object set to select from.  Must be a Tensor or list-like object.",
                }),
                "indexes_to_select": ("STRING", {
                    "tooltip":
                        "Indexes (zero-based) to select.  " +
                        "Supports ':' for range selection (including ':#', '#:' formats), '::#' & '#::#' & '#:#:#' formats for step selection, " +
                        "',' for multiple entries, and negative numbers for selecting from the end."
                    ,
                    "default": "0:",
                }),
                "multi_item_list": ("BOOLEAN", {
                    "tooltip":
                        "Indicates that the input is from an output node that supports list processing " +
                        "(usually marked with a multi-box icon).  If true, it will select from the initial given list, " +
                        "instead of selecting from each item in the list."
                    ,
                    "default": False,
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (IO.ANY, )
    RETURN_NAMES = ("any",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "The selected set.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Select specific indexes from a set, using a variety of supported syntaxes."

    CATEGORY = "LoadAnimAdv/util"

    def execute(self, any, indexes_to_select, multi_item_list):
        new_list = []

        if len(any) == 0:
            new_list = any
        elif multi_item_list[0]:
            new_list = select_indexes_from_any(obj=any, indexes_to_select=indexes_to_select[0])
        else:
            for i in any:
                new_list.append( select_indexes_from_any(obj=i, indexes_to_select=indexes_to_select[0]) )

        return new_list

class FlattenImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Image set to combine.",
                }),
                "masks": ("MASK", {
                    "tooltip": "Mask set to combine.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "The combined image set.",
        "The combined mask set.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Combines multiple sets of images (eg: from a multi-file load node) into a single set of frames, concatenating them in order."

    CATEGORY = "LoadAnimAdv/image"

    def execute(self, images=[], masks=[]):
        # Multiple items with INPUT_IS_LIST=True:
        #     images:    type=list size=2
        #     images[X]: type=Tensor size=torch.Size([75, 1024, 1024, 3])

        dest_images, dest_masks = None, None

        if len(images) == 0 and len(masks) == 0:
            print("FlattenImageList: No images or masks provided")
            return (None, None)

        if len(images) > 0:
            dest_images = torch.cat(tuple(images), dim=0)
        else:
            mask_size = masks[0].size()
            dest_images = torch.zeros((0, mask_size[1], mask_size[2], 3), dtype=torch.float32, device="cpu")

        if len(masks) > 0:
            dest_masks = torch.cat(tuple(masks), dim=0)
        else:
            image_size = images[0].size()
            dest_masks = torch.zeros((0, image_size[1], image_size[2], 1), dtype=torch.float32, device="cpu")

        return (dest_images, dest_masks,)

class AggregateNumberList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "numbers": ("NUMBER,FLOAT,INT", {
                    "tooltip":
                        "Number set to reduce.  Must have come from a node that supports list processing (usually marked with a multi-box icon), " +
                        "or was continued from a similar node in a chain."
                    ,
                }),
                "function": (['average', 'median', 'sum', 'prod', 'min', 'max', 'first', 'last'], {
                    "tooltip": "Function to apply to the numbers.",
                    "default": "average",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("FLOAT", "INT", "INT",)
    RETURN_NAMES = ("float", "int", "count",)
    OUTPUT_TOOLTIPS = (
        "The aggregated number, as a float.",
        "The aggregated number, as an integer.",
        "The count of numbers from the initial input.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Aggregates a list of numbers into a single number, using a variety of aggregation functions."

    CATEGORY = "LoadAnimAdv/util"

    def execute(self, numbers, function):
        if len(numbers) == 0:
            return (0.00, 0, 0,)
        elif function[0] == 'first':
            return (numbers[0], round(numbers[0]), len(numbers),)
        elif function[0] == 'last':
            return (numbers[-1], round(numbers[-1]), len(numbers),)

        func   = getattr(np, function[0])
        floats = np.array(numbers, dtype=float)
        result = func(floats)

        if isinstance(result, collections.abc.Sized):
            result = result[0]

        return (result, round(result), len(numbers),)

class FlattenAnyList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (IO.ANY, {
                    "tooltip": "Object set to combine.  Must be a list-like or iterable object.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (IO.ANY, )
    RETURN_NAMES = ("any",)
    OUTPUT_TOOLTIPS = (
        "The combined set.",
    )
    FUNCTION = "execute"
    DESCRIPTION = "Flattens a list of any type into a single set, concatenating them in order."

    CATEGORY = "LoadAnimAdv/util"

    def execute(self, any):
        if len(any) == 0:
            return (any,)

        dest = any[0]
        for i in any[1:]:
            if isinstance(dest, torch.Tensor):
                dest = torch.cat((dest, i), dim=0)
            elif isinstance(dest, collections.abc.Iterable):
                dest.extend(i)
            else:
                raise TypeError(f"Object is a '{type(dest).__qualname__}', not an iterable type.")

        return (dest,)

class BasicDimensionVariables:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "bus": ("BDV_BUS", {
                    "tooltip": "BDV bus input, from a previous BDV node.  If provided, will serve as an override for any zeroed inputs.",
                }),
                "width": ("INT", {
                    "tooltip": "Width of the image/video",
                    "default": 512,
                    "min": 0,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 16,
                }),
                "height": ("INT", {
                    "tooltip": "Height of the image/video",
                    "default": 512,
                    "min": 0,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 16,
                }),
                "frame_count": ("INT", {
                    "tooltip": "Frame length of the animation/video.  (Sometimes called 'length', in nodes like WanImageToVideo.)",
                    "default": 81,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                }),
                "batch_size": ("INT", {
                    "tooltip": "Batch size",
                    "default": 1,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                }),
                "fps": ("FLOAT", {
                    "tooltip": "Frames per second",
                    "default": 12.00,
                    "min": 0.00,
                    "max": 60.00,
                }),
                "seed": ("INT", {
                    "tooltip": "Random seed",
                    "default": 0x123456789abcdef0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            },
        }

    RETURN_TYPES = ("BDV_BUS", "INT", "INT", "INT", "INT", "FLOAT", "INT",)
    RETURN_NAMES = ("bus", "width", "height", "frame_count", "batch_size", "fps", "seed",)
    OUTPUT_TOOLTIPS = (
        "BDV bus output, for use in other BDV nodes.",
        "Width of the image/video",
        "Height of the image/video",
        "Frame length of the animation/video",
        "Batch size",
        "Frames per second",
        "Random seed",
    )
    FUNCTION = "execute"
    DESCRIPTION = (
        "Input node for declaring basic image/video dimensions, to be propagated to other nodes (such as KSampler).  " +
        "By having a single source for these values, you only have to change them once.  " +
        "The BDV Router can be used to route these values to other sections of the workflow."
    )

    CATEGORY = "LoadAnimAdv/util"

    def execute(self, bus=(0, 0, 0, 0, 0.00, 0), width=0, height=0, frame_count=0, batch_size=0, fps=0.00, seed=0):
        # Unpack bus, override, repack bus
        (bus_width, bus_height, bus_frame_count, bus_batch_size, bus_fps, bus_seed) = bus

        out_width       = width       if width       >= 0     else bus_width
        out_height      = height      if height      >= 0     else bus_height
        out_frame_count = frame_count if frame_count >= 0     else bus_frame_count
        out_batch_size  = batch_size  if batch_size  >= 0     else bus_batch_size
        out_fps         = fps         if fps         >= 0.00  else bus_fps
        out_seed        = seed        if seed        >= 0     else bus_seed

        out_fps = float(out_fps)
        out_bus = (out_width, out_height, out_frame_count, out_batch_size, out_fps, out_seed)

        return (out_bus, out_width, out_height, out_frame_count, out_batch_size, out_fps, out_seed,)

class BasicDimensionVariablesRouter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bus": ("BDV_BUS", {
                    "tooltip": "BDV bus input, from a previous BDV node.",
                }),
            },
        }

    RETURN_TYPES = ("BDV_BUS", "INT", "INT", "INT", "INT", "FLOAT", "INT",)
    RETURN_NAMES = ("bus", "width", "height", "frame_count", "batch_size", "fps", "seed",)
    OUTPUT_TOOLTIPS = (
        "BDV bus output, for use in other BDV nodes.",
        "Width of the image/video",
        "Height of the image/video",
        "Frame length of the animation/video",
        "Batch size",
        "Frames per second",
        "Random seed",
    )
    FUNCTION = "execute"
    DESCRIPTION = "A smaller bus-only 'Basic Dimension Variables' (BDV) node."
    CATEGORY = "LoadAnimAdv/util"

    def execute(self, bus=(0, 0, 0, 0, 0.00, 0)):
        (bus_width, bus_height, bus_frame_count, bus_batch_size, bus_fps, bus_seed) = bus
        return (bus, bus_width, bus_height, bus_frame_count, bus_batch_size, bus_fps, bus_seed,)

NODE_CLASS_MAPPINGS = {
    "LoadAnimAdv_LoadImageVideo":                LoadImageVideo,
    "LoadAnimAdv_LoadImageVideoFromPath":        LoadImageVideoFromPath,
    "LoadAnimAdv_LoadImagesVideosFromDirectory": LoadImagesVideosFromDirectory,
    "LoadAnimAdv_ListFilesFromDirectory":        ListFilesFromDirectory,
    "LoadAnimAdv_SelectIndexesFromImages":       SelectIndexesFromImages,
    "LoadAnimAdv_SelectIndexesFromAny":          SelectIndexesFromAny,
    "LoadAnimAdv_FlattenImageList":              FlattenImageList,
    "LoadAnimAdv_AggregateNumberList":           AggregateNumberList,
    "LoadAnimAdv_FlattenAnyList":                FlattenAnyList,
    "LoadAnimAdv_BasicDimensionVariables":       BasicDimensionVariables,
    "LoadAnimAdv_BasicDimensionVariablesRouter": BasicDimensionVariablesRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAnimAdv_LoadImageVideo":                "Load Image/Video",
    "LoadAnimAdv_LoadImageVideoFromPath":        "Load Image/Video From Path",
    "LoadAnimAdv_LoadImagesVideosFromDirectory": "Load Images/Videos From Directory",
    "LoadAnimAdv_ListFilesFromDirectory":        "List Files From Directory",
    "LoadAnimAdv_SelectIndexesFromImages":       "Select Indexes From Images",
    "LoadAnimAdv_SelectIndexesFromAny":          "Select Indexes From Any",
    "LoadAnimAdv_FlattenImageList":              "Flatten Image List",
    "LoadAnimAdv_AggregateNumberList":           "Aggregate Number List",
    "LoadAnimAdv_FlattenAnyList":                "Flatten Any List",
    "LoadAnimAdv_BasicDimensionVariables":       "Basic Dimension Variables",
    "LoadAnimAdv_BasicDimensionVariablesRouter": "BDV (Router)",
}
