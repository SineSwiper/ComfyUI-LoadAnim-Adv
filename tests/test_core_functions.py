import pytest
import os
import tempfile
import shutil
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, Mock, MagicMock

# Import the functions we're testing
try:
    from nodes import list_files_from_directory, load_image_video_from_path, select_indexes_from_any
except ImportError:
    # Fallback for when running tests directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from nodes import list_files_from_directory, load_image_video_from_path, select_indexes_from_any


class TestListFilesFromDirectory:
    """Test cases for list_files_from_directory function."""

    def test_basic_file_listing(self, test_files_directory):
        result = list_files_from_directory(
            directory=test_files_directory,
            file_list_cap=0,
            file_start_index=0,
            include_subfolders=False,
            valid_extensions='jpg,png',
            sort_method='None'
        )
        filenames = result[0]

        # Should find image files but not video or txt files
        assert len(filenames) >= 2
        assert any('image1.jpg' in f for f in filenames)
        assert any('image2.png' in f for f in filenames)
        assert not any('video.mp4' in f for f in filenames)
        assert not any('document.txt' in f for f in filenames)

    def test_include_subfolders(self, test_files_directory):
        result = list_files_from_directory(
            directory=test_files_directory,
            file_list_cap=0,
            file_start_index=0,
            include_subfolders=True,
            valid_extensions='jpg,png',
            sort_method='None'
        )
        filenames = result[0]

        # Should find files in subdirectories
        assert any('nested.jpg' in f for f in filenames)

    def test_file_list_cap(self, test_files_directory):
        result = list_files_from_directory(
            directory=test_files_directory,
            file_list_cap=1,
            file_start_index=0,
            include_subfolders=False,
            valid_extensions='jpg,png',
            sort_method='None'
        )
        filenames = result[0]

        # Should limit to 1 file
        assert len(filenames) == 1

    def test_file_start_index(self, test_files_directory):
        # First get all files
        all_result = list_files_from_directory(
            directory=test_files_directory,
            file_list_cap=0,
            file_start_index=0,
            include_subfolders=False,
            valid_extensions='jpg,png',
            sort_method='Alphabetical (ASC)'
        )
        all_files = all_result[0]

        if len(all_files) > 1:
            # Now get files starting from index 1
            result = list_files_from_directory(
                directory=test_files_directory,
                file_list_cap=0,
                file_start_index=1,
                include_subfolders=False,
                valid_extensions='jpg,png',
                sort_method='Alphabetical (ASC)'
            )
            subset_files = result[0]

            # Should be missing the first file
            assert len(subset_files) == len(all_files) - 1
            assert subset_files[0] == all_files[1]

    def test_valid_extensions_parsing(self, test_files_directory):
        # Test various extension formats
        result = list_files_from_directory(
            directory=test_files_directory,
            file_list_cap=0,
            file_start_index=0,
            include_subfolders=False,
            valid_extensions='jpg, png , mp4',  # With spaces
            sort_method='None'
        )
        filenames = result[0]

        # Should find jpg, png, and mp4 files
        extensions_found = set()
        for f in filenames:
            ext = f.split('.')[-1].lower()
            extensions_found.add(ext)

        expected = {'jpg', 'png', 'mp4'} & extensions_found
        assert len(expected) > 0

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError, match="Directory.*cannot be found"):
            list_files_from_directory(
                directory='/nonexistent/directory',
                file_list_cap=0,
                file_start_index=0,
                include_subfolders=False,
                valid_extensions='jpg,png',
                sort_method='None'
            )

    def test_empty_directory_no_matching_files(self, temp_dir):
        # Create directory with only non-matching files
        with open(os.path.join(temp_dir, 'test.txt'), 'w') as f:
            f.write('content')

        with pytest.raises(FileNotFoundError, match="No files in directory"):
            list_files_from_directory(
                directory=temp_dir,
                file_list_cap=0,
                file_start_index=0,
                include_subfolders=False,
                valid_extensions='jpg,png',
                sort_method='None'
            )

    def test_sorting_integration(self, test_files_directory):
        with patch('nodes.sort_by') as mock_sort:
            mock_sort.return_value = ['sorted', 'files']

            list_files_from_directory(
                directory=test_files_directory,
                file_list_cap=0,
                file_start_index=0,
                include_subfolders=False,
                valid_extensions='jpg,png',
                sort_method='Alphabetical (ASC)'
            )

            # Should call sort_by with the method
            mock_sort.assert_called_once()
            args = mock_sort.call_args[0]
            kwargs = mock_sort.call_args[1] if mock_sort.call_args[1] else {}

            assert kwargs.get('method') == 'Alphabetical (ASC)' or args[2] == 'Alphabetical (ASC)'


class TestLoadImageVideoFromPath:
    """Test cases for load_image_video_from_path function."""

    def create_test_image(self, path, mode='RGB', size=(100, 100), color='red'):
        """Helper to create test images."""
        img = Image.new(mode, size, color)
        img.save(path)
        return img

    def test_load_simple_rgb_image(self, temp_dir):
        image_path = os.path.join(temp_dir, 'test.png')
        self.create_test_image(image_path, mode='RGB', size=(50, 50), color=(255, 0, 0))

        with patch('nodes.node_helpers.pillow') as mock_pillow:
            mock_pillow.side_effect = lambda func, *args: func(*args)

            result = load_image_video_from_path(path=image_path, RGBA=False)
            output_image, output_mask, audio, frames, fps = result

            assert isinstance(output_image, torch.Tensor)
            assert output_image.shape[-1] == 3  # RGB channels
            assert frames == 1
            assert fps == 0.0
            assert audio is None

    def test_load_rgba_image_with_alpha(self, temp_dir):
        image_path = os.path.join(temp_dir, 'test.png')
        self.create_test_image(image_path, mode='RGBA', size=(50, 50), color=(255, 0, 0, 128))

        with patch('nodes.node_helpers.pillow') as mock_pillow:
            mock_pillow.side_effect = lambda func, *args: func(*args)

            result = load_image_video_from_path(path=image_path, RGBA=True)
            output_image, output_mask, audio, frames, fps = result

            assert isinstance(output_image, torch.Tensor)
            assert output_image.shape[-1] == 4  # RGBA channels
            assert isinstance(output_mask, torch.Tensor)

    def test_load_video_file(self, temp_dir):
        video_path = os.path.join(temp_dir, 'test.mp4')

        # Mock VideoFromFile
        mock_video = Mock()
        mock_components = Mock()
        mock_components.images = torch.rand(10, 100, 100, 3)
        mock_components.audio = torch.rand(1000, 2)
        mock_components.frame_rate = 30.0
        mock_video.get_components.return_value = mock_components

        with patch('nodes.VideoFromFile', return_value=mock_video):
            result = load_image_video_from_path(path=video_path)
            output_image, output_mask, audio, frames, fps = result

            assert isinstance(output_image, torch.Tensor)
            assert output_image.shape[0] == 10  # 10 frames
            assert fps == 30.0
            assert audio is not None

    def test_resize_functionality(self, temp_dir):
        image_path = os.path.join(temp_dir, 'test.png')
        self.create_test_image(image_path, size=(200, 100))

        with patch('nodes.node_helpers.pillow') as mock_pillow, \
             patch('nodes.resize_with_aspect_ratio') as mock_resize:

            mock_pillow.side_effect = lambda func, *args: func(*args)
            mock_resized = Image.new('RGB', (150, 150), 'blue')
            mock_resize.return_value = mock_resized

            load_image_video_from_path(
                path=image_path,
                width=150,
                height=150,
                keep_aspect_ratio='crop'
            )

            mock_resize.assert_called_once()
            args = mock_resize.call_args[0]
            assert args[1] == 150  # width
            assert args[2] == 150  # height
            assert args[3] == 'crop'  # keep_aspect_ratio

    def test_animated_gif_handling(self, temp_dir):
        # Create a simple animated GIF
        gif_path = os.path.join(temp_dir, 'test.gif')

        # Create frames
        frames = []
        for i in range(3):
            frame = Image.new('RGB', (50, 50), (i * 80, 0, 0))
            frame.info['duration'] = 100  # 100ms per frame
            frames.append(frame)

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100
        )

        with patch('nodes.node_helpers.pillow') as mock_pillow:
            mock_pillow.side_effect = lambda func, *args: func(*args)

            result = load_image_video_from_path(path=gif_path)
            output_image, output_mask, audio, frames_count, fps = result

            assert isinstance(output_image, torch.Tensor)
            assert frames_count == 3
            assert fps > 0  # Should calculate FPS from duration

    def test_mime_type_detection(self, temp_dir):
        # Test with unknown extension
        # Create a valid PNG first, then copy to unknown extension
        normal_path = os.path.join(temp_dir, 'test.png')
        weird_path = os.path.join(temp_dir, 'test.unknown')
        self.create_test_image(normal_path)

        # Copy the valid image to unknown extension
        import shutil
        shutil.copy2(normal_path, weird_path)

        with patch('mimetypes.guess_type', return_value=(None, None)), \
             patch('nodes.node_helpers.pillow') as mock_pillow:

            mock_pillow.side_effect = lambda func, *args: func(*args)

            # Should still work by falling back to image processing
            result = load_image_video_from_path(path=weird_path)
            output_image, output_mask, audio, frames, fps = result

            assert isinstance(output_image, torch.Tensor)


class TestSelectIndexesFromAny:
    """Test cases for select_indexes_from_any function."""

    def test_torch_tensor_single_index(self):
        tensor = torch.rand(5, 10, 10, 3)
        result = select_indexes_from_any(tensor, "2")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        assert torch.equal(result[0], tensor[2])

    def test_torch_tensor_multiple_indexes(self):
        tensor = torch.rand(5, 10, 10, 3)
        result = select_indexes_from_any(tensor, "0,2,4")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3
        assert torch.equal(result[0], tensor[0])
        assert torch.equal(result[1], tensor[2])
        assert torch.equal(result[2], tensor[4])

    def test_torch_tensor_range_selection(self):
        tensor = torch.rand(10, 5, 5, 3)
        result = select_indexes_from_any(tensor, "2:5")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # indexes 2, 3, 4
        assert torch.equal(result, tensor[2:5])

    def test_torch_tensor_open_range(self):
        tensor = torch.rand(5, 10, 10, 3)
        result = select_indexes_from_any(tensor, "2:")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # indexes 2, 3, 4
        assert torch.equal(result, tensor[2:])

    def test_torch_tensor_negative_indexes(self):
        tensor = torch.rand(5, 10, 10, 3)
        result = select_indexes_from_any(tensor, "-1")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        assert torch.equal(result[0], tensor[-1])

    def test_list_single_index(self):
        test_list = ['a', 'b', 'c', 'd', 'e']
        result = select_indexes_from_any(test_list, "2")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == 'c'

    def test_list_multiple_indexes(self):
        test_list = ['a', 'b', 'c', 'd', 'e']
        result = select_indexes_from_any(test_list, "0,2,4")

        assert isinstance(result, list)
        assert len(result) == 1  # itemgetter returns tuple in a list
        assert result[0] == ('a', 'c', 'e')

    def test_list_range_selection(self):
        test_list = ['a', 'b', 'c', 'd', 'e']
        result = select_indexes_from_any(test_list, "1:4")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ('b', 'c', 'd')

    def test_step_selection(self):
        tensor = torch.rand(10, 5, 5, 3)
        result = select_indexes_from_any(tensor, "0:6:2")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # indexes 0, 2, 4
        assert torch.equal(result[0], tensor[0])
        assert torch.equal(result[1], tensor[2])
        assert torch.equal(result[2], tensor[4])

    def test_complex_selection(self):
        tensor = torch.rand(10, 5, 5, 3)
        result = select_indexes_from_any(tensor, "0,2:5,7")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 5  # indexes 0, 2, 3, 4, 7

    def test_empty_selection_raises_error(self):
        tensor = torch.rand(5, 10, 10, 3)

        with pytest.raises(ValueError, match="No indexes selected"):
            select_indexes_from_any(tensor, "")

    def test_invalid_type_raises_error(self):
        invalid_obj = "not a tensor or list"

        with pytest.raises(TypeError, match="not a listable type"):
            select_indexes_from_any(invalid_obj, "0")

    def test_whitespace_handling(self):
        tensor = torch.rand(5, 10, 10, 3)
        result = select_indexes_from_any(tensor, " 0 , 2 : 4 ")

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # indexes 0, 2, 3
