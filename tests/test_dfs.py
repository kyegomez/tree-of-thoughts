# DFS

import pytest
from tree_of_thoughts import DFS


@pytest.fixture
def dfs_instance():
    return DFS()


def test_dfs_solve_default_parameters(dfs_instance, mocker):
    mocker.patch.object(
        dfs_instance.model,
        "generate_thoughts",
        return_value=["thought1", "thought2"],
    )
    mocker.patch.object(
        dfs_instance.model,
        "evaluate_states",
        return_value={"thought1": 1.0, "thought2": 0.6},
    )
    mocker.patch.object(
        dfs_instance.model, "generate_solution", return_value="final_solution"
    )

    solution = dfs_instance.solve(initial_prompt="initial", num_thoughts=2)
    assert solution == "final_solution"
