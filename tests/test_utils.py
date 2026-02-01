import pytest
import os
import tempfile
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, Mock

# Import the functions we're testing
import sys
try:
    from nodes import sort_by, resize_with_aspect_ratio, get_edge_color
except ImportError:
    # Fallback for when running tests directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from nodes import sort_by, resize_with_aspect_ratio, get_edge_color


class TestSortBy:
    """Test cases for the sort_by function."""

    def test_none_method_returns_original_order(self):
        items = ['zebra', 'apple', 'banana']
        result = sort_by(items, method='None')
        assert result == ['zebra', 'apple', 'banana']

    def test_alphabetical_asc(self):
        items = ['zebra', 'apple', 'banana']
        result = sort_by(items, method='Alphabetical (ASC)')
        assert result == ['apple', 'banana', 'zebra']

    def test_alphabetical_desc(self):
        items = ['zebra', 'apple', 'banana']
        result = sort_by(items, method='Alphabetical (DESC)')
        assert result == ['zebra', 'banana', 'apple']

    def test_sort_by_numerical_asc_mock_behavior(self):
        # Test numerical sorting by mocking the extract_first_number behavior
        with patch('nodes.extract_first_number') as mock_extract:
            mock_extract.side_effect = [1, 2, 10]
            items = ['file1.txt', 'file2.txt', 'file10.txt']

            result = sort_by(items, method='Numerical (ASC)')

            assert result == ['file1.txt', 'file2.txt', 'file10.txt']
            assert mock_extract.call_count == len(items)

    def test_sort_by_numerical_desc_mock_behavior(self):
        # Test numerical sorting by mocking the extract_first_number behavior
        with patch('nodes.extract_first_number') as mock_extract:
            mock_extract.side_effect = [1, 2, 10]
            items = ['file1.txt', 'file2.txt', 'file10.txt']

            result = sort_by(items, method='Numerical (DESC)')

            assert result == ['file10.txt', 'file2.txt', 'file1.txt']
            assert mock_extract.call_count == len(items)

    def test_datetime_sorting(self, temp_dir):
        # Create test files with different timestamps
        files = ['old.txt', 'new.txt']
        for i, filename in enumerate(files):
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write('content')

            # Set different modification times
            timestamp = 1000000000 + i * 86400  # Different days
            os.utime(filepath, (timestamp, timestamp))

        result = sort_by(files, base_path=temp_dir, method='Datetime (ASC)')
        assert result == ['old.txt', 'new.txt']

        result = sort_by(files, base_path=temp_dir, method='Datetime (DESC)')
        assert result == ['new.txt', 'old.txt']

    def test_datetime_with_missing_file(self, temp_dir):
        files = ['missing.txt', 'existing.txt']

        # Create only one file
        existing_path = os.path.join(temp_dir, 'existing.txt')
        with open(existing_path, 'w') as f:
            f.write('content')

        # Should handle missing file gracefully
        result = sort_by(files, base_path=temp_dir, method='Datetime (ASC)')
        assert 'missing.txt' in result
        assert 'existing.txt' in result

    def test_unknown_method_returns_original(self):
        items = ['zebra', 'apple', 'banana']
        result = sort_by(items, method='Unknown Method')
        assert result == ['zebra', 'apple', 'banana']

    def test_empty_list(self):
        result = sort_by([], method='Alphabetical (ASC)')
        assert result == []


class TestResizeWithAspectRatio:
    """Test cases for resize_with_aspect_ratio function."""

    @pytest.fixture
    def sample_square_image(self):
        return Image.new('RGB', (100, 100), color='red')

    @pytest.fixture
    def sample_wide_image(self):
        return Image.new('RGB', (200, 100), color='blue')

    @pytest.fixture
    def sample_tall_image(self):
        return Image.new('RGB', (100, 200), color='green')

    def test_stretch_mode(self, sample_square_image):
        result = resize_with_aspect_ratio(sample_square_image, 150, 75, 'stretch')
        assert result.size == (150, 75)

    def test_crop_mode_wide_image(self, sample_wide_image):
        # Wide image cropped to square should crop width
        result = resize_with_aspect_ratio(sample_wide_image, 100, 100, 'crop')
        assert result.size == (100, 100)

    def test_crop_mode_tall_image(self, sample_tall_image):
        # Tall image cropped to square should crop height
        result = resize_with_aspect_ratio(sample_tall_image, 100, 100, 'crop')
        assert result.size == (100, 100)

    def test_pad_mode_requires_edge_color(self, sample_wide_image):
        with patch('nodes.get_edge_color', return_value=(255, 0, 0, 255)):
            result = resize_with_aspect_ratio(sample_wide_image, 100, 200, 'pad')
            assert result.size == (100, 200)
            assert result.mode == 'RGBA'

    def test_pad_mode_wide_image_to_tall(self, sample_wide_image):
        with patch('nodes.get_edge_color', return_value=(255, 0, 0, 255)):
            # Wide image (2:1) padded to tall (1:2) should pad width
            result = resize_with_aspect_ratio(sample_wide_image, 50, 200, 'pad')
            assert result.size == (50, 200)

    def test_pad_mode_tall_image_to_wide(self, sample_tall_image):
        with patch('nodes.get_edge_color', return_value=(0, 255, 0, 255)):
            # Tall image (1:2) padded to wide (2:1) should pad height
            result = resize_with_aspect_ratio(sample_tall_image, 200, 50, 'pad')
            assert result.size == (200, 50)


class TestGetEdgeColor:
    """Test cases for get_edge_color function."""

    def test_get_edge_color_rgb(self):
        # Create image with solid red color
        img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        result = get_edge_color(img)

        # Should return RGBA tuple (converted from RGB)
        assert len(result) == 4
        assert result[0] == 255  # Red channel
        assert result[1] == 0    # Green channel
        assert result[2] == 0    # Blue channel

    def test_get_edge_color_rgba(self):
        # Create RGBA image with semi-transparent red
        img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
        result = get_edge_color(img)

        assert len(result) == 4
        assert result[0] == 255  # Red channel
        assert result[1] == 0    # Green channel
        assert result[2] == 0    # Blue channel
        assert result[3] == 128  # Alpha channel

    def test_get_edge_color_small_image(self):
        # Test with minimum size image
        img = Image.new('RGB', (1, 1), color=(100, 150, 200))
        result = get_edge_color(img)

        assert len(result) == 4
        assert result[0] == 100
        assert result[1] == 150
        assert result[2] == 200

    def test_get_edge_color_converts_to_rgba(self):
        # Test that function properly converts any mode to RGBA
        img = Image.new('L', (5, 5), color=128)  # Grayscale
        result = get_edge_color(img)

        # Should be converted to RGBA
        assert len(result) == 4
        # Grayscale should have equal RGB values
        assert result[0] == result[1] == result[2]


# Mock the extract_first_number function if it doesn't exist yet
def mock_extract_first_number(text):
    """Mock implementation of extract_first_number."""
    import re
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0


# Patch extract_first_number if it doesn't exist
if not hasattr(sys.modules['nodes'], 'extract_first_number'):
    sys.modules['nodes'].extract_first_number = mock_extract_first_number
