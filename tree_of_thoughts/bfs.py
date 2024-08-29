import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from tree_of_thoughts.agent import TotAgent

load_dotenv()


def string_to_dict(thought_string):
    return eval(thought_string)


class BFSWithTotAgent:
    """
    A class to perform Breadth-First Search (BFS) using the TotAgent, based on the ToT-BFS algorithm.

    Methods:
        bfs(state: str) -> Dict[str, Any]: Performs BFS and returns the final thought.
        run(task: str) -> str: Executes the BFS algorithm.
    """

    def __init__(
        self,
        agent: TotAgent,
        max_loops: int,
        breadth_limit: int,
        number_of_agents: int = 3,
        autosave_on: bool = True,
        id: str = uuid.uuid4().hex,
    ):
        """
        Initialize the BFSWithTotAgent class.

        Args:
            agent (TotAgent): An instance of the TotAgent class to generate and evaluate thoughts.
            max_loops (int): The maximum number of steps for the BFS algorithm.
            breadth_limit (int): The maximum number of states to consider at each level.
            number_of_agents (int): The number of thoughts to generate at each step. Default is 3.
            autosave_on (bool): Whether to save the results automatically. Default is True.
            id (str): A unique identifier for the BFS instance. Default is a randomly generated UUID.
        """
        self.id = id
        self.agent = agent
        self.max_loops = max_loops
        self.breadth_limit = breadth_limit
        self.number_of_agents = number_of_agents
        self.autosave_on = autosave_on
        self.all_thoughts = []  # Store all thoughts generated during BFS

    def bfs(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Perform Breadth-First Search (BFS) with a breadth limit based on evaluation scores.

        Args:
            state (str): The initial state or task to explore.

        Returns:
            Optional[Dict[str, Any]]: The final thought after the BFS completes.
        """
        # Initialize the set of states
        S = [state]

        for t in range(1, self.max_loops + 1):
            logger.info(f"Step {t}/{self.max_loops}: Expanding states.")

            # Generate new thoughts based on current states
            S_prime = self._generate_new_states(S)

            if not S_prime:  # If no new states were generated, stop the BFS
                logger.info(
                    f"No valid thoughts generated at step {t}. Stopping BFS."
                )
                break

            # Evaluate the new states
            V = self._evaluate_states(S_prime)

            # Log and store all thoughts
            self._log_and_store_thoughts(S_prime, V)

            # Select the best states based on their evaluations, limited by breadth_limit
            S = self._select_best_states(S_prime, V)

        # Return the best final thought
        return self._generate_final_answer(S)

    def _generate_new_states(self, S: List[str]) -> List[Dict[str, Any]]:
        """Generate new states (thoughts) from the current states."""
        S_prime = []
        for s in S:
            with ThreadPoolExecutor() as executor:
                new_thoughts = list(
                    executor.map(self.agent.run, [s] * self.number_of_agents)
                )
                S_prime.extend(
                    [
                        [s, thought]
                        for thought in new_thoughts
                        if thought is not None
                    ]
                )
        return S_prime

    def _evaluate_states(self, S_prime: List[Dict[str, Any]]) -> List[float]:
        """Evaluate the new states."""
        return [thought["evaluation"] for _, thought in S_prime]

    def _log_and_store_thoughts(
        self, S_prime: List[Dict[str, Any]], V: List[float]
    ):
        """Log and store all generated thoughts."""
        for i, (_, thought) in enumerate(S_prime):
            self.all_thoughts.append(thought)

    def _select_best_states(
        self, S_prime: List[Dict[str, Any]], V: List[float]
    ) -> List[str]:
        """Select the best states based on their evaluations, limited by breadth_limit."""
        # Pair states with their evaluations
        state_evaluation_pairs = list(zip(S_prime, V))

        # Sort based on evaluation scores
        state_evaluation_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select the top states based on the breadth limit
        best_states = [
            pair[0][1]["thought"]
            for pair in state_evaluation_pairs[: self.breadth_limit]
        ]
        return best_states

    def _generate_final_answer(self, S: List[str]) -> Optional[Dict[str, Any]]:
        """Generate the final answer by selecting the best thought from the last set of states."""
        if not S:
            return None
        final_state = max(S, key=lambda s: self.agent.run(s)["evaluation"])
        return self.agent.run(final_state)

    def _run_agent(self, task: str) -> Optional[Dict[str, Any]]:
        """Run the agent to generate a thought and its evaluation."""
        try:
            agent_output = self.agent.run(task)

            return string_to_dict(agent_output)
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
        return None

    def run(self, task: str) -> str:
        """
        Execute the BFS algorithm.

        Args:
            task (str): The initial task or state to start the BFS.

        Returns:
            str: The final thought after BFS as a JSON string.
        """
        final_thought = self.bfs(task)

        # Sort all thoughts by evaluation score in ascending order
        self.all_thoughts.sort(key=lambda x: x["evaluation"], reverse=False)

        # Prepare JSON structure for logging all thoughts
        tree_dict = {
            "all_thoughts": self.all_thoughts,
            "final_thought": final_thought,
        }

        # Save the results if autosave is enabled
        # if self.autosave_on:
        #     _save_dict_to_json(tree_dict, self.id)

        return json.dumps(tree_dict, indent=4)
