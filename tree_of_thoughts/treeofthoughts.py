import concurrent.futures
import json
import logging
import os
import time
from queue import PriorityQueue
from typing import Any, Dict, Union

import numpy as np


DATA_PATH = "./data"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TreeofThoughts:
    """
    A class representing a tree of thoughts.

    Attributes:
        model: The model used for evaluation.
        tree: The tree structure containing the nodes and their evaluations.
        best_state: The best state found so far.
        best_value: The best value found so far.
        history: The history of evaluated states.

    Methods:
        save_tree_to_json: Saves the tree structure to a JSON file.
        logNewState: Logs a new state and its evaluation.
        adjust_pruning_threshold_precentile: Adjusts the pruning threshold based on the evaluated thoughts using percentile.
        adjust_pruning_threshold_moving_average: Adjusts the pruning threshold based on the evaluated thoughts using moving average.
    """

    def __init__(self, model):
        self.model = model
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = []  # added line initalize history

    def save_tree_to_json(self, file_name):
        """
        Saves the tree structure to a JSON file.

        Args:
            file_name: The name of the JSON file to save the tree structure to.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(self.tree, json_file, indent=4)

    def logNewState(self, state, evaluation):
        """
        Logs a new state and its evaluation.

        Args:
            state: The state to log.
            evaluation: The evaluation of the state.
        """
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree["nodes"]:
            self.tree["nodes"][state]["thoughts"].append(evaluation)
        else:
            self.tree["nodes"][state] = {"thoughts": [evaluation]}

    def adjust_pruning_threshold_precentile(
        self, evaluated_thoughts, percentile
    ):
        """
        Adjusts the pruning threshold based on the evaluated thoughts using percentile.

        Args:
            evaluated_thoughts: A dictionary of evaluated thoughts.
            percentile: The percentile value to use for adjusting the threshold.

        Returns:
            The adjusted pruning threshold.
        """
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0
        return max(np.percentile(values, percentile), 0.1)

    def adjust_pruning_threshold_moving_average(
        self, evaluated_thoughts, window_size
    ):
        """
        Adjusts the pruning threshold based on the evaluated thoughts using moving average.

        Args:
            evaluated_thoughts: A dictionary of evaluated thoughts.
            window_size: The size of the moving average window.

        Returns:
            The adjusted pruning threshold.
        """
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)


######################


class BFS(TreeofThoughts):
    def solve(
        self,
        initial_prompt,
        num_thoughts,
        max_steps,
        max_states,
        value_threshold,
        pruning_threshold=0.5,
    ):
        current_states = [initial_prompt]
        state_values = {}
        dynamic_pruning_threshold = pruning_threshold

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for step in range(1, max_steps + 1):
                    selected_states = []
                    for state in current_states:
                        thoughts = self.model.generate_thoughts(
                            state, num_thoughts, initial_prompt
                        )
                        futures = [
                            executor.submit(
                                self.model.evaluate_states,
                                {thought: 0},
                                initial_prompt,
                            )
                            for thought in thoughts
                        ]
                        concurrent.futures.wait(futures)
                        evaluated_thoughts = {
                            thought: fut.result()
                            for thought, fut in zip(thoughts, futures)
                            if isinstance(fut.result(), (int, float))
                        }  # check if result is a number

                        if (
                            evaluated_thoughts
                        ):  # only adjust if you have evaluated thoughts
                            dynamic_pruning_threshold = (
                                self.adjust_pruning_threshold_moving_average(
                                    evaluated_thoughts, 5
                                )
                            )

                        for thought, value in evaluated_thoughts.items():
                            flattened_state = (
                                (state, thought)
                                if isinstance(state, str)
                                else (*state, thought)
                            )
                            selected_states.append((flattened_state, value))

                        selected_states.sort(key=lambda x: x[1], reverse=True)
                        selected_states = selected_states[
                            :max_states
                        ]  # Select only the top states

                        for state, value in selected_states:
                            if value >= dynamic_pruning_threshold:
                                state_values[state] = value
                                self.logNewState(state, value)
                                logger.debug(f"State Values: {state_values}")

            # if state_values:
            #     highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
            #     print(f"highest rated solution: {highest_rated_solution}")
            #     highest_rated_state = highest_rated_solution[0]  # Use a different name to avoid confusion
            #     print(f'highest rated state: {highest_rated_state}')
            #     try:
            #         solution = self.model.generate_solution(initial_prompt, highest_rated_state)
            #     except Exception as e:
            #         logger.error(f"Error in generating solution: {e}")
            #         solution = None  # Set a fallback value for solution

            #     return solution if solution is not None else highest_rated_state  # Return highest rated state if solution is None
            if state_values:
                highest_rated_solution = max(
                    state_values.items(), key=lambda x: x[1]
                )
                highest_rated_state = highest_rated_solution[0]
                solution = self.model.generate_solution(
                    initial_prompt, highest_rated_state
                )
                print(
                    "Highest_rated solution:"
                    f" {highest_rated_solution} highest_rated_solution:"
                    f" {highest_rated_solution} Solution: {solution}"
                )

                return solution if solution else highest_rated_state

            else:
                return None

        except Exception as e:
            logger.error(f"Error in tot_bfs: {e}")
            return None


###########


class DFS(TreeofThoughts):
    """
    Depth-first search implementation for the TreeofThoughts class.

    Args:
        TreeofThoughts (class): Base class for the TreeofThoughtsDFS class.

    Methods:
        solve: Solves the problem using depth-first search algorithm.

    Attributes:
        None.
    """

    def solve(
        self,
        initial_prompt,
        num_thoughts,
        max_steps,
        value_threshold,
        pruning_threshold=0.5,
    ):
        output = []

        def dfs(state, step):
            nonlocal output
            if step > max_steps:
                thought = self.model.generate_thoughts(state, 1, initial_prompt)
                value = self.model.evaluate_states({state}, initial_prompt)[
                    state
                ]
                output.append((thought, value))
                return

            thoughts = self.model.generate_thoughts(
                state, self.num_thoughts, initial_prompt
            )
            evaluated_thoughts = self.model.evaluate_states(
                {thought: 0 for thought in thoughts}, initial_prompt
            )
            filtered_thoughts = [
                thought
                for thought in thoughts
                if evaluated_thoughts[thought] >= self.pruning_threshold
            ]

            for next_state in filtered_thoughts:
                state_value = self.model.evaluate_states(
                    {next_state: 0}, initial_prompt
                )[next_state]

                if state_value > self.value_threshold:
                    child = (
                        (state, next_state)
                        if isinstance(state, str)
                        else (*state, next_state)
                    )
                    dfs(child, step + 1)

        try:
            dfs(initial_prompt, 1)
            best_state, _ = max(output, key=lambda x: x[1])
            solution = self.model.generate_solution(initial_prompt, best_state)
            return solution if solution else best_state
        except Exception as e:
            logger.error(f"Error in tot_dfs: {e}")
            return None


# v2 => best first search => explores state space of the quality of the states
# priority que or greedy BFS
class BESTSearch(TreeofThoughts):
    """
    Represents a tree of thoughts.

    Attributes:
        model: The model used for generating and evaluating thoughts.
        tree: The tree structure to store the thoughts and evaluations.
    """

    def __init__(self, model):
        """
        Initializes a TreeofThoughtsBEST object.

        Args:
            model: The model used for generating and evaluating thoughts.
        """
        self.model = model
        self.tree = {"nodes": {}}

    def save_tree_to_json(self, file_name):
        """
        Saves the tree structure to a JSON file.

        Args:
            file_name: The name of the JSON file to save the tree structure to.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(self.tree, json_file, indent=4)

    def log_new_state(self, state, evaluation):
        """
        Logs a new state and its evaluation in the tree structure.

        Args:
            state: The state to log.
            evaluation: The evaluation of the state.
        """
        state_key = " | ".join(state) if isinstance(state, tuple) else state
        if state_key in self.tree["nodes"]:
            self.tree["nodes"][state_key]["thoughts"].append(evaluation)
        else:
            self.tree["nodes"]["state_key"] = {"thoughts": [evaluation]}

    def solve(self, initial_prompt, num_thoughts, max_steps, pruning_threshold):
        """
        Solves the tree of thoughts problem.

        Args:
            initial_prompt: The initial prompt for generating thoughts.
            num_thoughts: The number of thoughts to generate at each step.
            max_steps: The maximum number of steps to perform.
            pruning_threshold: The threshold for pruning thoughts.

        Returns:
            The solution to the tree of thoughts problem.
        """
        visited_states = set()
        state_queue = PriorityQueue()

        state_queue.put((0, initial_prompt))

        for _ in range(max_steps):
            if state_queue.empty():
                break

            _, state = state_queue.get()

            if state in visited_states:
                continue

            visited_states.add(state)

            thoughts = self.model.generate_thoughts(
                state, num_thoughts, initial_prompt
            )
            evaluated_thoughts = {
                thought: self.model.evaluate_states(
                    {thought: 0}, initial_prompt
                )[thought]
                for thought in thoughts
            }

            for thought, value in evaluated_thoughts.items():
                if value >= pruning_threshold:
                    new_state = (
                        (state, thought)
                        if isinstance(state, str)
                        else (*state, thought)
                    )
                    state_queue.put((value, new_state))
                    self.log_new_state(new_state, value)

        best_state = max(visited_states, key=self.model.evaluate_states)
        solution = self.model.generate_solution(initial_prompt, best_state)
        print(f"Highest_rated solution: {best_state}  Solution: {solution}")
        return solution if solution else best_state


# A* search algorithm
class ASearch:
    def __init__(self, model):
        self.model = model

    def solve(
        self,
        initial_prompt,
        num_thoughts=5,
        max_steps=30,
        pruning_threshold=0.4,
    ):
        # the open set is implemented as a piorituve quue where the priority is -f_score
        open_set = PriorityQueue()
        open_set.put((0, 0, initial_prompt))

        # the set of visited_states
        visited_states = set()

        # the g_scores and f-scores are stored as dictionaries
        g_scores = {initial_prompt: 0}
        f_scores = {
            initial_prompt: self.model.evaluate_states(
                {initial_prompt: 0}, initial_prompt
            )[initial_prompt]
        }

        # the parent of each state is stored in a dictionary
        came_from = {}

        for _ in range(max_steps):
            if open_set.empty():
                break

            _, _, current_state = open_set.get()

            if self.is_goal(current_state, f_scores[current_state]):
                return self.reconstruct_path(
                    came_from, current_state, initial_prompt
                )

            thoughts = self.model.generate_thoughts(
                current_state, num_thoughts, initial_prompt
            )
            evaluated_thoughts = {
                thought: self.model.evaluate_states(
                    {thought: 0}, initial_prompt
                )[thought]
                for thought in thoughts
            }

            for thought, value in evaluated_thoughts.items():
                if value < pruning_threshold or thought in visited_states:
                    continue

                tentative_g_score = g_scores[current_state] + 1 / value
                if (
                    thought not in g_scores
                    or tentative_g_score < g_scores[thought]
                ):
                    came_from[thought] = current_state
                    g_scores[thought] = tentative_g_score
                    f_scores[thought] = tentative_g_score + value
                    open_set.put(
                        (-f_scores[thought], g_scores[thought], thought)
                    )

        return self.reconstruct_path(came_from, current_state, initial_prompt)

    def is_goal(self, state, score):
        # if eval state is above 0.9
        return score >= 0.9

    def reconstruct_path(self, came_from, current_state, initial_prompt):
        path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        path.reverse()

        path = self.reconstruct_path(came_from, current_state, initial_prompt)
        solution = self.model.generate_solution(initial_prompt, path)
        print(f"Path: {path} solution: {solution}")
        return solution if solution else path


class MonteCarloSearch(TreeofThoughts):
    """
    A class representing a Monte Carlo Tree of Thoughts.

    Attributes:
        model (Model): The model used for generating thoughts and evaluating states.
        objective (str): The objective of the optimization process.
        solution_found (bool): Indicates whether a solution has been found.
        tree (Dict[str, Dict[str, Union[float, Dict[str, Any]]]]): The tree structure containing nodes, thoughts, and evaluations.

    Methods:
        __init__(self, model, objective="balance"): Initializes a MonteCarloSearch instance.
        optimize_params(self, num_thoughts, max_steps, max_states): Optimizes the parameters based on the objective.
        solve(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold): Solves the problem using Monte Carlo search.
        monte_carlo_search(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold): Performs the Monte Carlo search algorithm.
    """

    def __init__(self, model, objective="balance"):
        super().__init__(model)
        self.objective = objective
        self.solution_found = False
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
            "metrics": {"thoughts": {}, "evaluations": {}},
        }

    def optimize_params(self, num_thoughts, max_steps, max_states):
        """
        Optimizes the parameters based on the objective.

        Args:
            num_thoughts (int): The number of thoughts to generate.
            max_steps (int): The maximum number of steps in the search.
            max_states (int): The maximum number of states to consider.

        Returns:
            Tuple[int, int, int]: The optimized values of num_thoughts, max_steps, and max_states.
        """
        if self.objective == "speed":
            num_thoughts = max(1, num_thoughts - 1)
            max_steps = max(1, max_steps - 1)
            max_states = max(1, max_states - 1)
        elif self.objective == "reliability":
            num_thoughts += 1
            max_steps += 1
            max_states += 1
        elif self.objective == "balance":
            if self.solution_found:
                num_thoughts = max(1, num_thoughts - 1)
                max_steps = max(1, max_steps - 1)
                max_states = max(1, max_states - 1)
            else:
                num_thoughts += 1
                max_steps += 1
                max_states += 1

        return num_thoughts, max_steps, max_states

    def solve(
        self,
        initial_prompt: str,
        num_thoughts: int,
        max_steps: int,
        max_states: int,
        pruning_threshold: float,
    ):
        """
        Solves the problem using Monte Carlo search.

        Args:
            initial_prompt (str): The initial prompt for the search.
            num_thoughts (int): The number of thoughts to generate.
            max_steps (int): The maximum number of steps in the search.
            max_states (int): The maximum number of states to consider.
            pruning_threshold (float): The threshold for pruning states.

        Returns:
            Union[str, Tuple]: The solution generated by the model or the best state found.
        """
        self.file_name = "logs/tree_of_thoughts_output_montecarlo.json"
        return self.monte_carlo_search(
            initial_prompt,
            num_thoughts,
            max_steps,
            max_states,
            pruning_threshold,
        )

    def monte_carlo_search(
        self,
        initial_prompt: str,
        num_thoughts: int,
        max_steps: int,
        max_states: int,
        pruning_threshold: float,
    ):
        """
        Performs the Monte Carlo search algorithm.

        Args:
            initial_prompt (str): The initial prompt for the search.
            num_thoughts (int): The number of thoughts to generate.
            max_steps (int): The maximum number of steps in the search.
            max_states (int): The maximum number of states to consider.
            pruning_threshold (float): The threshold for pruning states.

        Returns:
            Union[str, Tuple]: The solution generated by the model or the best state found.
        """
        current_states = [initial_prompt]
        state_values = {}
        visit_counts = {initial_prompt: 0}
        transposition_table = {}

        best_state = None
        best_value = float("-inf")

        for step in range(1, max_steps + 1):
            selected_states = []

            for state in current_states:
                if state in transposition_table:
                    transposition_table[state]
                else:
                    time.sleep(1)
                    thoughts = self.model.generate_thoughts(
                        state, num_thoughts, initial_prompt
                    )
                    time.sleep(1)
                    evaluated_thoughts = self.model.evaluate_states(
                        thoughts, initial_prompt
                    )

                    for thought, value in evaluated_thoughts.items():
                        flattened_state = (
                            (state, thought)
                            if isinstance(state, str)
                            else (*state, thought)
                        )
                        transposition_table[flattened_state] = value

                for thought, value in evaluated_thoughts.items():
                    flattened_state = (
                        (state, thought)
                        if isinstance(state, str)
                        else (*state, thought)
                    )

                    if flattened_state not in visit_counts:
                        visit_counts[flattened_state] = 0

                    if (
                        visit_counts[state] > visit_counts[flattened_state]
                        and visit_counts[flattened_state] > 0
                    ):
                        ucb1_value = value + np.sqrt(
                            2
                            * np.log(visit_counts[state])
                            / visit_counts[flattened_state]
                        )

                        if ucb1_value >= pruning_threshold:
                            selected_states.append(flattened_state)
                            state_values[flattened_state] = value

                            # Update the best state if the current state value is greater than the best value
                            if value > best_value:
                                best_state = flattened_state
                                best_value = value

                visit_counts[state] += 1

            if len(selected_states) > max_states:
                current_states = selected_states[:max_states]
            self.save_tree_to_json(self.file_name)

        # if best_state is not None:
        #     solution = self.model.generate_solution(initial_prompt, best_state)
        #     return solution
        # else:
        #     solution = None

        # return None
        solution = self.model.generate_solution(initial_prompt, best_state)
        return solution if solution else best_state
