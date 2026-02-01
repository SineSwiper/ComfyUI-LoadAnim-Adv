import pytest
import os
import sys
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch

# Add parent directory to path to import nodes
root_loadanim_adv_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, root_loadanim_adv_dir)

# Add ComfyUI path for dependencies like node_helpers
comfyui_path = None
for starting_path in [
    root_loadanim_adv_dir,
    os.path.dirname(sys.argv[0]),
]:
    path = os.path.abspath(starting_path)
    while path != '/':
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'comfyui_version.py')):
            comfyui_path = path
            break
        path = os.path.dirname(path)
    if comfyui_path:
        break

if comfyui_path:
    sys.path.insert(0, comfyui_path)
else:
    raise Exception("Cannot find ComfyUI root for testing")

import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    return img

@pytest.fixture
def sample_image_rgba():
    """Create a sample RGBA PIL Image for testing."""
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    return img

@pytest.fixture
def sample_torch_image():
    """Create a sample torch tensor image for testing."""
    # Shape: (batch, height, width, channels)
    return torch.rand(1, 100, 100, 3, dtype=torch.float32)

@pytest.fixture
def sample_torch_image_batch():
    """Create a sample batch of torch tensor images for testing."""
    # Shape: (batch, height, width, channels)
    return torch.rand(5, 100, 100, 3, dtype=torch.float32)

@pytest.fixture
def sample_mask():
    """Create a sample torch mask for testing."""
    return torch.ones(100, 100, dtype=torch.float32)

@pytest.fixture
def mock_folder_paths():
    """Mock the folder_paths module."""
    with patch('folder_paths.get_input_directory') as mock_input_dir, \
         patch('folder_paths.get_annotated_filepath') as mock_annotated, \
         patch('folder_paths.filter_files_content_types') as mock_filter:

        mock_input_dir.return_value = '/mock/input'
        mock_annotated.side_effect = lambda x: f'/mock/input/{x}'
        mock_filter.return_value = ['test.jpg', 'test.png']

        yield {
            'input_dir': mock_input_dir,
            'annotated': mock_annotated,
            'filter': mock_filter
        }

@pytest.fixture
def mock_video_component():
    """Mock video component for testing."""
    mock_component = Mock()
    mock_component.images = torch.rand(10, 100, 100, 3)
    mock_component.audio = torch.rand(1000, 2)
    mock_component.frame_rate = 30.0
    return mock_component

@pytest.fixture
def test_files_directory(temp_dir):
    """Create a test directory with sample files."""
    # Create subdirectories
    subdir = os.path.join(temp_dir, 'subdir')
    os.makedirs(subdir)

    # Create test files
    test_files = [
        'image1.jpg',
        'image2.png',
        'image10.jpg',
        'video.mp4',
        'document.txt',
        os.path.join('subdir', 'nested.jpg')
    ]

    for filename in test_files:
        filepath = os.path.join(temp_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write('test content')

    return temp_dir
