import pytest
from tree_of_thoughts.tot_agent import ToTAgent
from swarms import Agent
import unittest


class TestToTAgent:
    @pytest.fixture
    def mock_agent(self, monkeypatch):
        # Mocking the Agent class
        class MockAgent:
            def __call__(self, task):
                return f"mocked response for {task}"

        monkeypatch.setattr("tree_of_thoughts.tot_agent.Agent", MockAgent)

    def test_initialization_default(self, mock_agent):
        totagent = ToTAgent(Agent())
        assert totagent.agent is not None
        assert totagent.strategy == "cot"
        assert totagent.enable_react is True
        assert totagent.k == 3

    def test_run_method_call_count(self, mock_agent):
        totagent = ToTAgent(Agent())
        task = "test task"
        with unittest.mock.patch.object(totagent, "agent") as mocked_agent:
            mocked_agent.return_value = "response"
            thoughts = totagent.run(task)
            assert mocked_agent.call_count == totagent.k
            assert len(thoughts) == totagent.k

    # ... more test cases
