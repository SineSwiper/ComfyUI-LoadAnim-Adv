import pytest
import os
import torch
import numpy as np
from unittest.mock import patch, Mock

# Import the node classes we're testing
try:
    from nodes import (
        SelectIndexesFromAny, FlattenImageList, AggregateNumberList,
        FlattenAnyList, BasicDimensionVariables, BasicDimensionVariablesRouter
    )
except ImportError:
    # Fallback for when running tests directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from nodes import (
        SelectIndexesFromAny, FlattenImageList, AggregateNumberList,
        FlattenAnyList, BasicDimensionVariables, BasicDimensionVariablesRouter
    )

# Import the IO type
try:
    from comfy.comfy_types.node_typing import IO
except ImportError:
    # Mock IO if not available
    class IO:
        ANY = "ANY"


class TestSelectIndexesFromAny:
    """Test cases for SelectIndexesFromAny node."""

    def test_input_types_structure(self):
        input_types = SelectIndexesFromAny.INPUT_TYPES()

        assert 'required' in input_types
        assert 'any' in input_types['required']
        assert 'indexes_to_select' in input_types['required']
        assert 'multi_item_list' in input_types['required']

    def test_return_types(self):
        assert SelectIndexesFromAny.RETURN_TYPES == (IO.ANY,)
        assert SelectIndexesFromAny.RETURN_NAMES == ("any",)
        assert SelectIndexesFromAny.OUTPUT_IS_LIST == (True,)
        assert SelectIndexesFromAny.INPUT_IS_LIST == True

    @patch('nodes.select_indexes_from_any')
    def test_execute_multi_item_list_false(self, mock_select):
        mock_select.side_effect = [
            torch.rand(1, 10, 10, 3),
            torch.rand(2, 10, 10, 3)
        ]

        node = SelectIndexesFromAny()
        input_list = [
            torch.rand(5, 10, 10, 3),
            torch.rand(3, 10, 10, 3)
        ]

        result = node.execute(
            any=input_list,
            indexes_to_select=["0", "0:2"],
            multi_item_list=[False, False]
        )

        assert len(result) == 2
        assert mock_select.call_count == 2

    def test_execute_empty_list(self):
        node = SelectIndexesFromAny()
        result = node.execute(
            any=[],
            indexes_to_select=["0"],
            multi_item_list=[False]
        )

        assert result == []


class TestFlattenImageList:
    """Test cases for FlattenImageList node."""

    def test_input_types_structure(self):
        input_types = FlattenImageList.INPUT_TYPES()

        assert 'optional' in input_types
        assert 'images' in input_types['optional']
        assert 'masks' in input_types['optional']

    def test_return_types(self):
        assert FlattenImageList.RETURN_TYPES == ("IMAGE", "MASK")
        assert FlattenImageList.RETURN_NAMES == ("images", "masks")
        assert FlattenImageList.INPUT_IS_LIST == True

    def test_execute_with_images_and_masks(self):
        node = FlattenImageList()
        images = [
            torch.rand(2, 100, 100, 3),
            torch.rand(3, 100, 100, 3)
        ]
        masks = [
            torch.rand(2, 100, 100),
            torch.rand(3, 100, 100)
        ]

        result = node.execute(images=images, masks=masks)

        assert len(result) == 2
        assert result[0].shape[0] == 5  # 2 + 3 frames
        assert result[1].shape[0] == 5  # 2 + 3 masks

    def test_execute_with_images_only(self):
        node = FlattenImageList()
        images = [
            torch.rand(2, 100, 100, 3),
            torch.rand(1, 100, 100, 3)
        ]

        result = node.execute(images=images, masks=[])

        assert len(result) == 2
        assert result[0].shape[0] == 3  # 2 + 1 frames
        assert result[1].shape[0] == 0  # Empty mask tensor

    def test_execute_empty_inputs(self):
        node = FlattenImageList()
        result = node.execute(images=[], masks=[])

        assert result == (None, None)


class TestAggregateNumberList:
    """Test cases for AggregateNumberList node."""

    def test_input_types_structure(self):
        input_types = AggregateNumberList.INPUT_TYPES()

        assert 'required' in input_types
        assert 'numbers' in input_types['required']
        assert 'function' in input_types['required']

    def test_return_types(self):
        assert AggregateNumberList.RETURN_TYPES == ("FLOAT", "INT", "INT")
        assert AggregateNumberList.RETURN_NAMES == ("float", "int", "count")
        assert AggregateNumberList.INPUT_IS_LIST == True

    def test_execute_average(self):
        node = AggregateNumberList()
        numbers = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = node.execute(numbers=numbers, function=['average'])

        assert result[0] == 3.0  # float result
        assert result[1] == 3    # int result
        assert result[2] == 5    # count

    def test_execute_sum(self):
        node = AggregateNumberList()
        numbers = [1.0, 2.0, 3.0]

        result = node.execute(numbers=numbers, function=['sum'])

        assert result[0] == 6.0
        assert result[1] == 6
        assert result[2] == 3

    def test_execute_first_last(self):
        node = AggregateNumberList()
        numbers = [10.0, 20.0, 30.0]

        result_first = node.execute(numbers=numbers, function=['first'])
        result_last = node.execute(numbers=numbers, function=['last'])

        assert result_first[0] == 10.0
        assert result_last[0] == 30.0

    def test_execute_empty_list(self):
        node = AggregateNumberList()
        result = node.execute(numbers=[], function=['average'])

        assert result == (0.0, 0, 0)


class TestFlattenAnyList:
    """Test cases for FlattenAnyList node."""

    def test_input_types_structure(self):
        input_types = FlattenAnyList.INPUT_TYPES()

        assert 'required' in input_types
        assert 'any' in input_types['required']

    def test_return_types(self):
        assert FlattenAnyList.RETURN_TYPES == (IO.ANY,)
        assert FlattenAnyList.RETURN_NAMES == ("any",)
        assert FlattenAnyList.INPUT_IS_LIST == True

    def test_execute_with_tensors(self):
        node = FlattenAnyList()
        tensor_list = [
            torch.rand(2, 10, 10, 3),
            torch.rand(3, 10, 10, 3)
        ]

        result = node.execute(any=tensor_list)

        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[0] == 5  # 2 + 3 frames

    def test_execute_with_lists(self):
        node = FlattenAnyList()
        list_of_lists = [
            [1, 2, 3],
            [4, 5],
            [6]
        ]

        result = node.execute(any=list_of_lists)

        assert len(result) == 1
        assert result[0] == [1, 2, 3, 4, 5, 6]

    def test_execute_empty_input(self):
        node = FlattenAnyList()
        result = node.execute(any=[])

        assert result == ([],)


class TestBasicDimensionVariables:
    """Test cases for BasicDimensionVariables node."""

    def test_input_types_structure(self):
        input_types = BasicDimensionVariables.INPUT_TYPES()

        assert 'required' in input_types
        assert 'optional' in input_types
        assert 'bus' in input_types['optional']
        assert 'width' in input_types['optional']
        assert 'height' in input_types['optional']

    def test_return_types(self):
        expected_types = ("BDV_BUS", "INT", "INT", "INT", "INT", "FLOAT", "INT")
        expected_names = ("bus", "width", "height", "frame_count", "batch_size", "fps", "seed")

        assert BasicDimensionVariables.RETURN_TYPES == expected_types
        assert BasicDimensionVariables.RETURN_NAMES == expected_names

    def test_execute_without_bus(self):
        node = BasicDimensionVariables()
        result = node.execute(
            width=1024,
            height=768,
            frame_count=60,
            batch_size=4,
            fps=24.0,
            seed=12345
        )

        bus, width, height, frame_count, batch_size, fps, seed = result

        assert width == 1024
        assert height == 768
        assert frame_count == 60
        assert batch_size == 4
        assert fps == 24.0
        assert seed == 12345
        assert bus == (1024, 768, 60, 4, 24.0, 12345)

    def test_execute_with_bus_override(self):
        node = BasicDimensionVariables()
        input_bus = (512, 512, 30, 2, 15.0, 54321)

        result = node.execute(
            bus=input_bus,
            width=0,  # Should use bus value
            height=1024,  # Should override bus value
            frame_count=0,  # Should use bus value
            batch_size=8,  # Should override bus value
            fps=0.0,  # Should use bus value
            seed=0  # Should use bus value
        )

        bus, width, height, frame_count, batch_size, fps, seed = result

        assert width == 512   # From bus
        assert height == 1024 # Override
        assert frame_count == 30  # From bus
        assert batch_size == 8    # Override
        assert fps == 15.0    # From bus
        assert seed == 54321  # From bus


class TestBasicDimensionVariablesRouter:
    """Test cases for BasicDimensionVariablesRouter node."""

    def test_input_types_structure(self):
        input_types = BasicDimensionVariablesRouter.INPUT_TYPES()

        assert 'required' in input_types
        assert 'bus' in input_types['required']

    def test_return_types(self):
        expected_types = ("BDV_BUS", "INT", "INT", "INT", "INT", "FLOAT", "INT")
        expected_names = ("bus", "width", "height", "frame_count", "batch_size", "fps", "seed")

        assert BasicDimensionVariablesRouter.RETURN_TYPES == expected_types
        assert BasicDimensionVariablesRouter.RETURN_NAMES == expected_names

    def test_execute(self):
        node = BasicDimensionVariablesRouter()
        input_bus = (1920, 1080, 120, 8, 30.0, 42)

        result = node.execute(bus=input_bus)

        bus, width, height, frame_count, batch_size, fps, seed = result

        assert bus == input_bus
        assert width == 1920
        assert height == 1080
        assert frame_count == 120
        assert batch_size == 8
        assert fps == 30.0
        assert seed == 42

    def test_execute_default_bus(self):
        node = BasicDimensionVariablesRouter()

        result = node.execute()  # Should use default bus

        bus, width, height, frame_count, batch_size, fps, seed = result

        assert bus == (0, 0, 0, 0, 0.0, 0)
        assert width == 0
        assert height == 0
        assert frame_count == 0
        assert batch_size == 0
        assert fps == 0.0
        assert seed == 0
