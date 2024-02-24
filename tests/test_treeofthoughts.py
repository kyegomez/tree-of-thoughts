import pytest
import numpy as np
import os
import json
from treeofthoughts.tree_of_thoughts import TreeofThoughts


# Sample model fixture
@pytest.fixture
def sample_model():
    # Pretend we have a model fixture
    return "model"


# Sample state fixture
@pytest.fixture
def sample_state():
    return "initial state"


# Sample evaluation fixture
@pytest.fixture
def sample_evaluation():
    return 0.5


# Sample evaluated thoughts fixture
@pytest.fixture
def sample_evaluated_thoughts():
    return {"state1": 0.1, "state2": 0.2, "state3": 0.3}


@pytest.fixture
def test_tree(sample_model):
    # Initializing an instance of TreeofThoughts with the sample model
    return TreeofThoughts(sample_model)


# Actual tests
class TestTreeofThoughts:
    def test_initialization(self, sample_model):
        # Check that the class is initialized correctly
        tree = TreeofThoughts(sample_model)
        assert tree.model == sample_model
        assert isinstance(tree.tree, dict)
        assert tree.best_state is None
        assert tree.best_value == float("-inf")
        assert tree.history == []

    def test_save_tree_to_json(self, test_tree, tmp_path):
        # Note: tmp_path is a pytest fixture that provides a temporary directory unique to the test invocation
        file_name = tmp_path / "tree.json"
        test_tree.save_tree_to_json(str(file_name))
        assert os.path.exists(file_name)
        with open(file_name) as f:
            data = json.load(f)
        assert data == test_tree.tree

    def test_log_new_state(self, test_tree, sample_state, sample_evaluation):
        # First log, node doesn't exist yet
        test_tree.log_new_state(sample_state, sample_evaluation)
        assert sample_state in test_tree.tree["nodes"]
        assert test_tree.tree["nodes"][sample_state]["thoughts"] == [
            sample_evaluation
        ]

        # Second log, same state: checking if the evaluation is appended
        test_tree.log_new_state(sample_state, sample_evaluation)
        assert len(test_tree.tree["nodes"][sample_state]["thoughts"]) == 2

    def test_adjust_pruning_threshold_percentile(
        self, test_tree, sample_evaluated_thoughts
    ):
        expected_threshold = np.percentile(
            list(sample_evaluated_thoughts.values()), 90
        )
        actual_threshold = test_tree.adjust_pruning_threshold_precentile(
            sample_evaluated_thoughts, 90
        )
        assert actual_threshold == expected_threshold

    def test_adjust_pruning_threshold_moving_average(
        self, test_tree, sample_evaluated_thoughts
    ):
        window_size = 2
        values = list(sample_evaluated_thoughts.values())
        expected_threshold = np.mean(values[-window_size:])
        actual_threshold = test_tree.adjust_pruning_threshold_moving_average(
            sample_evaluated_thoughts, window_size
        )
        assert actual_threshold == expected_threshold


# Add more test cases with various input combinations to test edge cases
@pytest.mark.parametrize(
    "state,evaluation", [("state1", 0.1), ("state2", 0.5), ("state3", 0.9)]
)
def test_log_new_state_with_multi_inputs(test_tree, state, evaluation):
    # Testing log_new_state with multiple inputs
    test_tree.log_new_state(state, evaluation)
    assert state in test_tree.tree["nodes"]
    assert evaluation in test_tree.tree["nodes"][state]["thoughts"]
