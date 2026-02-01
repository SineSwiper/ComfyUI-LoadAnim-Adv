import pytest
import os
import torch
import numpy as np
from unittest.mock import patch, Mock, MagicMock

# Import the node classes we're testing
try:
    from nodes import (
        LoadImageVideo, LoadImageVideoFromPath, ListFilesFromDirectory,
        LoadImagesVideosFromDirectory, SelectIndexesFromImages
    )
except ImportError:
    # Fallback for when running tests directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from nodes import (
        LoadImageVideo, LoadImageVideoFromPath, ListFilesFromDirectory,
        LoadImagesVideosFromDirectory, SelectIndexesFromImages
    )


class TestLoadImageVideo:
    """Test cases for LoadImageVideo node."""

    def test_input_types_structure(self):
        input_types = LoadImageVideo.INPUT_TYPES()

        assert 'required' in input_types
        assert 'image' in input_types['required']
        assert 'RGBA' in input_types['required']
        assert 'width' in input_types['required']
        assert 'height' in input_types['required']
        assert 'keep_aspect_ratio' in input_types['required']

    def test_return_types(self):
        assert LoadImageVideo.RETURN_TYPES == ("IMAGE", "MASK", "AUDIO", "INT", "FLOAT")
        assert LoadImageVideo.RETURN_NAMES == ("images", "mask", "audio", "frame_count", "fps")

    @patch('folder_paths.get_annotated_filepath')
    @patch('nodes.load_image_video_from_path')
    def test_execute(self, mock_load_func, mock_get_path):
        mock_get_path.return_value = '/mock/path/test.jpg'
        mock_load_func.return_value = (
            torch.rand(1, 100, 100, 3),  # images
            torch.rand(1, 100, 100),     # mask
            None,                        # audio
            1,                           # frame_count
            0.0                          # fps
        )

        node = LoadImageVideo()
        result = node.execute(
            image='test.jpg',
            RGBA=False,
            width=100,
            height=100,
            keep_aspect_ratio='crop'
        )

        mock_get_path.assert_called_once_with('test.jpg')
        mock_load_func.assert_called_once_with(
            path='/mock/path/test.jpg',
            RGBA=False,
            width=100,
            height=100,
            keep_aspect_ratio='crop'
        )

        assert len(result) == 5
        assert isinstance(result[0], torch.Tensor)


class TestLoadImageVideoFromPath:
    """Test cases for LoadImageVideoFromPath node."""

    def test_input_types_structure(self):
        input_types = LoadImageVideoFromPath.INPUT_TYPES()

        assert 'required' in input_types
        assert 'path' in input_types['required']
        assert input_types['required']['path'][0] == "STRING"

    @patch('nodes.load_image_video_from_path')
    def test_execute(self, mock_load_func):
        mock_load_func.return_value = (
            torch.rand(1, 50, 50, 3),
            torch.rand(1, 50, 50),
            None,
            1,
            0.0
        )

        node = LoadImageVideoFromPath()
        result = node.execute(
            path='/test/path.jpg',
            RGBA=True,
            width=50,
            height=50,
            keep_aspect_ratio='pad'
        )

        mock_load_func.assert_called_once_with(
            path='/test/path.jpg',
            RGBA=True,
            width=50,
            height=50,
            keep_aspect_ratio='pad'
        )

        assert len(result) == 5


class TestListFilesFromDirectory:
    """Test cases for ListFilesFromDirectory node."""

    def test_input_types_structure(self):
        input_types = ListFilesFromDirectory.INPUT_TYPES()

        assert 'required' in input_types
        assert 'directory' in input_types['required']
        assert 'optional' in input_types
        assert 'file_list_cap' in input_types['optional']
        assert 'sort_method' in input_types['optional']

    def test_return_types(self):
        assert ListFilesFromDirectory.RETURN_TYPES == ("STRING",)
        assert ListFilesFromDirectory.RETURN_NAMES == ("filenames",)
        assert ListFilesFromDirectory.OUTPUT_IS_LIST == (True,)

    @patch('nodes.list_files_from_directory')
    def test_execute(self, mock_list_func):
        mock_list_func.return_value = (['file1.jpg', 'file2.png'],)

        node = ListFilesFromDirectory()
        result = node.execute(
            directory='/test/dir',
            file_list_cap=10,
            file_start_index=0,
            include_subfolders=True,
            valid_extensions='jpg,png',
            sort_method='Alphabetical (ASC)'
        )

        mock_list_func.assert_called_once_with(
            directory='/test/dir',
            file_list_cap=10,
            file_start_index=0,
            include_subfolders=True,
            valid_extensions='jpg,png',
            sort_method='Alphabetical (ASC)'
        )

        assert result == (['file1.jpg', 'file2.png'],)


class TestLoadImagesVideosFromDirectory:
    """Test cases for LoadImagesVideosFromDirectory node."""

    def test_input_types_structure(self):
        input_types = LoadImagesVideosFromDirectory.INPUT_TYPES()

        assert 'required' in input_types
        assert 'directory' in input_types['required']
        assert 'frame_indexes_to_select' in input_types['required']
        assert 'flatten_frames' in input_types['required']

    def test_return_types(self):
        expected_types = ("IMAGE", "MASK", "AUDIO", "INT", "FLOAT", "STRING")
        expected_names = ("image", "mask", "audio", "frame_counts", "fps", "image_path")
        expected_list = (True, True, True, True, True, True)

        assert LoadImagesVideosFromDirectory.RETURN_TYPES == expected_types
        assert LoadImagesVideosFromDirectory.RETURN_NAMES == expected_names
        assert LoadImagesVideosFromDirectory.OUTPUT_IS_LIST == expected_list

    @patch('nodes.list_files_from_directory')
    @patch('nodes.load_image_video_from_path')
    @patch('nodes.select_indexes_from_any')
    def test_execute_flatten_frames_false(self, mock_select, mock_load, mock_list):
        # Mock file listing
        mock_list.return_value = (['file1.jpg', 'file2.jpg'],)

        # Mock loading images
        mock_load.side_effect = [
            (torch.rand(2, 100, 100, 3), torch.rand(2, 100, 100), None, 2, 15.0),
            (torch.rand(3, 100, 100, 3), torch.rand(3, 100, 100), None, 3, 12.0)
        ]

        # Mock frame selection
        mock_select.side_effect = [
            torch.rand(1, 100, 100, 3),  # selected frames from first image
            torch.rand(1, 100, 100),     # selected mask from first image
            torch.rand(2, 100, 100, 3),  # selected frames from second image
            torch.rand(2, 100, 100),     # selected mask from second image
        ]

        node = LoadImagesVideosFromDirectory()
        result = node.execute(
            directory='/test/dir',
            RGBA=False,
            width=100,
            height=100,
            keep_aspect_ratio='crop',
            image_file_load_cap=0,
            image_file_start_index=0,
            frame_indexes_to_select='0',
            total_frame_load_cap=0,
            flatten_frames=False,
            include_subfolders=False,
            valid_extensions='jpg,png',
            sort_method='None'
        )

        images, masks, audios, frame_counts, fps_list, image_paths = result

        assert len(images) == 2
        assert len(masks) == 2
        assert len(image_paths) == 2
        assert 'file1.jpg' in image_paths
        assert 'file2.jpg' in image_paths


class TestSelectIndexesFromImages:
    """Test cases for SelectIndexesFromImages node."""

    def test_input_types_structure(self):
        input_types = SelectIndexesFromImages.INPUT_TYPES()

        assert 'required' in input_types
        assert 'images' in input_types['required']
        assert 'indexes_to_select' in input_types['required']

    def test_return_types(self):
        assert SelectIndexesFromImages.RETURN_TYPES == ("IMAGE",)
        assert SelectIndexesFromImages.RETURN_NAMES == ("images",)

    @patch('nodes.select_indexes_from_any')
    def test_execute_with_tensor(self, mock_select):
        mock_select.return_value = torch.rand(2, 100, 100, 3)

        node = SelectIndexesFromImages()
        input_images = torch.rand(5, 100, 100, 3)

        result = node.execute(input_images, "0,2")

        mock_select.assert_called_once_with(obj=input_images, indexes_to_select="0,2")
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    def test_execute_with_empty_list(self):
        node = SelectIndexesFromImages()
        result = node.execute([], "0")

        assert result == ([],)
