#thought -> evaluated value (0.4, This solution is invalid because x) -> thought prompt + this solution is invalid because + better eval

import os
import time
import json
DATA_PATH = './data'
import logging 
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from typing import Any, Dict, Union
import re
import numpy as np
import concurrent.futures
from queue import PriorityQueue








# class TreeofThoughts:
#     def __init__(self, model, search_algorithm):
#         self.model = model
#         self.search_algorithm = search_algorithm
#         self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
#             "nodes": {},
#         }
#         self.best_state = None
#         self.best_value = float("-inf")
#         self.history = [] #added line initalize history

#     def save_tree_to_json(self, file_name):
#         os.makedirs(os.path.dirname(file_name), exist_ok=True)
#         with open(file_name, 'w') as json_file:
#             json.dump(self.tree, json_file, indent=4)

#     def logNewState(self, state, evaluation):
#         if not (type(state) == str):
#             state = " | ".join(state)
#         if state in self.tree["nodes"]:
#             self.tree["nodes"][state]['thoughts'].append(evaluation)
#         else:
#             self.tree["nodes"][state] = {'thoughts': [evaluation]} 


#     def adjust_pruning_threshold_percentile(self, evaluated_thoughts, percentile):
#         values = np.array(list(evaluated_thoughts.values()))
#         if values.size == 0:
#             return 0
#         return max(np.percentile(values, percentile), 0.1)  # Add a minimum threshold of 0.1

#     def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
#         values = list(evaluated_thoughts.values())
#         if len(values) < window_size:
#             return np.mean(values) if values else 0 
#         else:
#             return max(np.mean(values[-window_size:]), 0.1)  # Add a minimum threshold of 0.1

            
#     def tot_bfs(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold):
#         current_states = [initial_prompt]
#         state_values = {}
#         dynamic_pruning_threshold = pruning_threshold

#         try:
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 for step in range(1, max_steps + 1):
#                     selected_states = []
#                     for state in current_states:
#                         thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
#                         futures = [executor.submit(self.model.evaluate_states, {thought: 0}, initial_prompt) for thought in thoughts]
#                         concurrent.futures.wait(futures)
#                         evaluated_thoughts = {thought: fut.result() for thought, fut in zip(thoughts, futures) if isinstance(fut.result(), (int, float))} # check if result is a number
                        
#                         if evaluated_thoughts: # only adjust if you have evaluated thoughts 
#                             # Adjust dynamic pruning threshold, choose one of the following
#                             # dynamic_pruning_threshold = self.adjust_pruning_threshold_percentile(evaluated_thoughts, 30)
#                             dynamic_pruning_threshold = self.adjust_pruning_threshold_moving_average(evaluated_thoughts, 5)

#                         for thought, value in evaluated_thoughts.items():
#                             flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
#                             selected_states.append((flattened_state, value))

#                         # Sort and select top states
#                         selected_states.sort(key=lambda x: x[1], reverse=True)
#                         selected_states = selected_states[:max_states]  # Select only the top states

#                         for state, value in selected_states:
#                             if value >= dynamic_pruning_threshold:
#                                 state_values[state] = value
#                                 self.logNewState(state, value)
#                                 logger.debug(f"State Values: {state_values}")

#             # Always select best state from state_values if available
#             if state_values:
#                 highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
#                 print(f"highest rated solution: {highest_rated_solution}")
#                 best_state = highest_rated_solution[0]
#                 print(f'best state: {best_state}')
#                 try:
#                     # solution = self.model.generate_solution(initial_prompt, best_state)
#                     return best_state
#                 except Exception as e:
#                     logger.error(f"Error in generating solution: {e}")
#                     return None
#             else:
#                 return None

#         except Exception as e:
#             logger.error(f"Error in tot_bfs: {e}")
#             return None



#     def tot_dfs(self, initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold):
#         output = []

#         def dfs(state, step):
#             nonlocal output
#             if step > max_steps:
#                 thought = self.model.generate_thoughts(state, 1, initial_prompt)
#                 value = self.model.evaluate_states({state}, initial_prompt)[state]
#                 output.append((thought, value))
#                 return 

#             thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
#             filtered_thoughts = self.filter_thoughts(thoughts)

#             for next_state in filtered_thoughts:
#                 state_value = self.model.evaluate_states({next_state}, initial_prompt)[next_state]
#                 logger.info(f"state: {next_state}, value: {state_value}")

#                 if state_value > value_threshold and state_value >= pruning_threshold:
#                     child = (state, next_state) if isinstance(state, str) else (*state, next_state)
#                     dfs(child, step + 1)

#         try:
#             dfs(initial_prompt, 1)
#             best_state = max(output, key=lambda x: x[1])
#             return best_state[0]
#         except Exception as e:
#             logger.error(f"Error in tot_dfs: {e}")
#             return None

#     def solve(self, initial_prompt, num_thoughts, max_steps, max_states, value_threshold, pruning_threshold=0.5):
#         self.file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"

#         try:
#             if self.search_algorithm == 'BFS':
#                 result = self.tot_bfs(initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     return result

#             elif self.search_algorithm == 'DFS':
#                 result = self.tot_dfs(initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     return result

#             else:
#                 raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
            
#             if result is not None:
#                 logger.info("No solution found, returning best evaluated thought so far")
#                 result = self.best_state
#             self.tree["solution"] = result

#         except KeyboardInterrupt:
#             logger.error("Keyboard interrupt detected.")
#         except ValueError as e:
#             logger.error(f"Error: {e}")
#         finally:
#             logger.info("Saving the current tree and metrics.")
#             self.save_tree_to_json(self.file_name)

#         return result


class TreeofThoughts:
    def __init__(self, model):
        self.model = model
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = [] #added line initalize history


    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def logNewState(self, state, evaluation):
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree['nodes']:
            self.tree['nodes'][state]['thoughts'].append(evaluation)
        else:
            self.tree['nodes'][state] = {'thoughts': [evaluation]}

    def adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile):
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0 
        return max(np.percentile(values, percentile), 0.1)
    

    def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)



######################  

class TreeofThoughtsBFS(TreeofThoughts):
    def solve(self, initial_prompt, num_thoughts, max_steps, max_states, value_threshold, pruning_threshold=0.5):
        current_states = [initial_prompt]
        state_values = {}
        dynamic_pruning_threshold = pruning_threshold

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for step in range(1, max_steps + 1):
                    selected_states = []
                    for state in current_states:
                        thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
                        futures = [executor.submit(self.model.evaluate_states, {thought: 0}, initial_prompt) for thought in thoughts]
                        concurrent.futures.wait(futures)
                        evaluated_thoughts = {thought: fut.result() for thought, fut in zip(thoughts, futures) if isinstance(fut.result(), (int, float))}  # check if result is a number

                        if evaluated_thoughts:  # only adjust if you have evaluated thoughts
                            dynamic_pruning_threshold = self.adjust_pruning_threshold_moving_average(evaluated_thoughts, 5)

                        for thought, value in evaluated_thoughts.items():
                            flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
                            selected_states.append((flattened_state, value))

                        selected_states.sort(key=lambda x: x[1], reverse=True)
                        selected_states = selected_states[:max_states]  # Select only the top states

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
                highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
                highest_rated_state = highest_rated_solution[0]  
                solution = self.model.generate_solution(initial_prompt, highest_rated_state)
                print(f"Highest_rated solution: {highest_rated_solution} highest_rated_solution: {highest_rated_solution} Solution: {solution}")

                return solution if solution else highest_rated_state

            else:
                return None

        except Exception as e:
            logger.error(f"Error in tot_bfs: {e}")
            return None


###########

class TreeofThoughtsDFS(TreeofThoughts):
    def solve(self, initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold=0.5):
        output = []

        def dfs(state, step):
            nonlocal output
            if step > max_steps:
                thought = self.model.generate_thoughts(state, 1, initial_prompt)
                value = self.model.evaluate_states({state}, initial_prompt)[state]
                output.append((thought, value))
                return

            thoughts = self.model.generate_thoughts(state, self.num_thoughts, initial_prompt)
            evaluated_thoughts = self.model.evaluate_states({thought: 0 for thought in thoughts}, initial_prompt)
            filtered_thoughts = [thought for thought in thoughts if evaluated_thoughts[thought] >= self.pruning_threshold]


            for next_state in filtered_thoughts:
                state_value = self.model.evaluate_states({next_state: 0}, initial_prompt)[next_state]

                if state_value > self.value_threshold:
                    child = (state, next_state) if isinstance(state, str) else (*state, next_state)
                    dfs(child, step + 1)
        try:
            dfs(initial_prompt, 1)
            best_state, _ = max(output, key=lambda x: x[1])
            solution = self.model.generate_solution(initial_prompt, best_state)
            return solution if solution else best_state
        except Exception as e:
            logger.error(f"Error in tot_dfs: {e}")
            return None


#v2 => best first search => explores state space of the quality of the states
#priority que or greedy BFS
class TreeofThoughtsBEST:
    def __init__(self, model):
        self.model = model
        self.tree = {"nodes": {}}

    def save_tree_to_json(self, file_name):
        os.makdirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def log_new_state(self, state, evaluation):
        state_key = " | ".join(state) if isinstance(state, tuple) else state
        if state_key in self.tree["nodes"]:
            self.tree["nodes"][state_key]['thoughts'].append(evaluation)
        else:
            self.tree['nodes']['state_key'] = {'thoughts': [evaluation]}

    def solve(self, initial_prompt, num_thoughts, max_steps, pruning_threshold):
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

            thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
            evaluated_thoughts = {thought: self.model.evaluate_states({thought: 0}, initial_prompt)[thought] for thought in thoughts}

            for thought, value in evaluated_thoughts.items():
                if value >= pruning_threshold:
                    new_state = (state, thought) if isinstance(state, str) else (*state, thought)
                    state_queue.put((value, new_state))
                    self.log_new_state(new_state, value)
        
        best_state = max(visited_states, key=self.model.evaluate_states)
        solution = self.model.generate_solution(initial_prompt, best_state)
        print(f"Highest_rated solution: {best_state}  Solution: {solution}")
        return solution if solution else best_state

#A* search algorithm
class TreeofThoughtsASearch:
    def __init__(self, model):
        self.model = model

    def solve(self, initial_prompt, num_thoughts=5, max_steps=30, pruning_threshold=0.4):
        #the open set is implemented as a piorituve quue where the priority is -f_score
        open_set = PriorityQueue()
        open_set.put((0, 0, initial_prompt))

        #the set of visited_states
        visited_states = set()


        #the g_scores and f-scores are stored as dictionaries
        g_scores = {initial_prompt: 0}
        f_scores = {initial_prompt: self.model.evaluate_states({initial_prompt: 0}, initial_prompt)[initial_prompt]}


        #the parent of each state is stored in a dictionary
        came_from = {}

        for _ in range(max_steps):
            if open_set.empty():
                break

            _, _, current_state = open_set.get()

            if self.is_goal(current_state, f_scores[current_state]):
                return self.reconstruct_path(came_from, current_state, initial_prompt)

            thoughts = self.model.generate_thoughts(current_state, num_thoughts, initial_prompt)
            evaluated_thoughts = {thought: self.model.evaluate_states({thought: 0}, initial_prompt)[thought] for thought in thoughts}

            for thought, value in evaluated_thoughts.items():
                if value < pruning_threshold or thought in visited_states:
                    continue

                tentative_g_score = g_scores[current_state] + 1 / value
                if thought not in g_scores or tentative_g_score < g_scores[thought]:
                    came_from[thought] = current_state
                    g_scores[thought] = tentative_g_score
                    f_scores[thought] = tentative_g_score + value
                    open_set.put((-f_scores[thought], g_scores[thought], thought))

        return self.reconstruct_path(came_from, current_state, initial_prompt)

    
    def is_goal(self, state, score):
        #if eval state is above 0.9
        return score >= 0.9
    
    def reconstruct_path(self, came_from, current_state, initial_prompt):
        path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        path.reverse()

        path = self.reconstruct_path(came_from, current_state)
        solution = self.model.generate_solution(initial_prompt, path)
        print(f"Path: {path} solution: {solution}")
        return solution if solution else path


class MonteCarloTreeofThoughts(TreeofThoughts):
    def __init__(self, model, objective="balance"):
        super().__init__(model)
        self.objective = objective
        self.solution_found = False
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
            "metrics": {"thoughts": {}, "evaluations": {}},
        }


    def optimize_params(self, num_thoughts, max_steps, max_states):
        if self.objective == 'speed':
            num_thoughts = max(1, num_thoughts - 1)
            max_steps = max(1, max_steps - 1)
            max_states = max(1, max_states - 1)
        elif self.objective == 'reliability':
            num_thoughts += 1
            max_steps += 1
            max_states += 1
        elif self.objective == 'balanace':
            if self.solution_found:
                num_thoughts = max(1, num_thoughts - 1)
                max_steps = max(1, max_steps - 1)
                max_states = max(1, max_states - 1)
            else:
                num_thoughts += 1
                max_steps += 1
                max_states += 1
        
        return num_thoughts, max_steps, max_states

    def solve(self,
              initial_prompt: str,
              num_thoughts: int,
              max_steps: int,
              max_states: int,
              pruning_threshold: float,
            #   sleep_time: float,
              ):
        file_name = str(initial_prompt)
        self.file_name = f"logs/tree_of_thoughts_output_{file_name}.json"
        return self.monte_carlo_search(
            initial_prompt,
            num_thoughts,
            max_steps,
            max_states,
            pruning_threshold,
            # sleep_time,
        )
#v3
    def monte_carlo_search(self,
                        initial_prompt: str,
                        num_thoughts: int,
                        max_steps: int,
                        max_states: int,
                        pruning_threshold: float,
                        ):
        current_states = [initial_prompt]
        state_values = {}
        visit_counts = {initial_prompt: 0}
        transposition_table = {}

        best_state = None
        best_value = float('-inf')

        for step in range(1, max_steps + 1):
            selected_states = []

            for state in current_states:
                if state in transposition_table:
                    state_value = transposition_table[state]
                else:
                    time.sleep(1)
                    thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
                    time.sleep(1)
                    evaluated_thoughts = self.model.evaluate_states(thoughts, initial_prompt)

                    for thought, value in evaluated_thoughts.items():
                        flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
                        transposition_table[flattened_state] = value

                for thought, value in evaluated_thoughts.items():
                    flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)

                    if flattened_state not in visit_counts:
                        visit_counts[flattened_state] = 0

                    if visit_counts[state] > visit_counts[flattened_state] and visit_counts[flattened_state] > 0:
                        ucb1_value = value + np.sqrt(2 * np.log(visit_counts[state]) / visit_counts[flattened_state])

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

#does not output state after each thought --- idk why -- needs work
class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        print(f'Start time {start_time}')
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b, pruning_threshold=0.5)
                print(f'result in optimized tree of thoughts: {result}')
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth, confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")



# if __name__ == '__main__':
    
#     #create instance
#     parser = argparse.ArgumentParser(description="Tree of Thoughts Solver")
#     parser.add_argument("--problem", type=str, required=True, help="Initial problem statement")
#     parser.add_argument("--version", type=int, choices=[1, 2], default=1, help="Version of Tree of Thoughts to use (v1 or v2)")


#     # input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

#     # parser.add_argument("--problem", type=str, required=True, help="Initial problem statement")
#     parser.add_argument("--search_algorithm", type=str, choices=["BFS", "DFS"], default="BFS", help="Search algorithm to use (BFS or DFS)")
#     parser.add_argument("--k", type=int, default=3, help="Number of thoughts to generate")
#     parser.add_argument("--T", type=int, default=10, help="Step limit")
#     parser.add_argument("--b", type=int, default=5, help="Number of most promising states")
#     parser.add_argument("--vth", type=float, default=0.4, help="Value threshold for DFS")
#     parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds before stopping")
#     parser.add_argument("--confidence", type=float, default=0.8, help="Model confidence threshold")
#     parser.add_argument("--max_iterations", type=int, default=40, help="Maximum number of tree branch nodes")
#     parser.add_argument("--convergence_threshold", type=float, default=0.01, help="Convergence threshold for the search process")
#     parser.add_argument("--convergence_count", type=int, default=5, help="Number of searches to be considered converged")







# Yes, you're right. In the current implementation of MonteCarloTreeofThoughts, the search ends prematurely as soon as a solution is found in the current step. This can be an issue, especially when we aim to find the most reliable (best) solution in the shortest amount of time.

# There are several potential improvements that can be made:

# Exploration vs Exploitation: In the current algorithm, all thoughts (states) generated in each step are treated equally. However, we can apply the concept of exploration vs exploitation by prioritizing the exploration of promising states (thoughts). This could be done by sorting the states based on their evaluation values and exploring the higher ranked states first.

# Iterative Deepening: We can incorporate the concept of iterative deepening into the algorithm. This means that we start the search with a shallow depth limit and incrementally increase the depth limit in each iteration until a solution is found or the depth limit exceeds max_steps. This allows us to explore a larger portion of the search space and have a chance to find a better solution.

# Asynchronous Evaluation: Currently, the algorithm waits until all states are evaluated before proceeding to the next step. We can potentially speed up the process by asynchronously evaluating the states and updating the best solution whenever a better solution is found.

# Below is the updated class implementing the above improvements:

# python
# Copy code
# from typing import Dict, Union, Any, List
# from multiprocessing import Pool

# class MonteCarloTreeofThoughts(TreeofThoughts):
#     def __init__(self, model, objective='balance'):
#         super().__init__(model, "MonteCarlo")
#         self.objective = objective
#         self.solution_found = False
#         self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
#             "nodes": {},
#             "metrics": {"thoughts": {}, "evaluations": {}},
#         }
#         self.pool = Pool()  # For parallel evaluation

#     # other methods ...

#     def monte_carlo_search(self,
#                            initial_prompt: str,
#                            num_thoughts: int,
#                            max_steps: int,
#                            max_states: int,
#                            pruning_threshold: float):
#         current_states = [initial_prompt]
#         state_values = {}
#         best_state = None
#         best_value = float('-inf')

#         for step in range(1, max_steps + 1):
#             selected_states = []
#             for state in current_states:
#                 thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
#                 evaluated_thoughts = self.pool.map(self.model.evaluate_states, [(thought, initial_prompt) for thought in thoughts])
#                 evaluated_thoughts = dict(evaluated_thoughts)

#                 self.tree["metrics"]["thoughts"].update({thought: state for thought in thoughts})
#                 self.tree["metrics"]["evaluations"].update(evaluated_thoughts)

#                 # Sort the evaluated thoughts by their values
#                 evaluated_thoughts = dict(sorted(evaluated_thoughts.items(), key=lambda item: item[1], reverse=True))

#                 for thought, value in evaluated_thoughts.items():
#                     if value >= pruning_threshold:
#                         flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
#                         selected_states.append(flattened_state)
#                         state_values[flattened_state] = value
#                         self.logNewState(flattened_state, value)

#                         # Update the best state
#                         if value > best_value:
#                             best_state = flattened_state
#                             best_value = value
#                             self.solution_found = True
#             if len(selected_states) > max_states:
#                 current_states = selected_states[:max_states]

#             num_thought




# class TreeofThoughts:
#     def __init__(self, model, search_algorithm):
#         self.model = model
#         self.search_algorithm = search_algorithm
#         self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
#             "nodes": {},
#             # "rejected_paths": {}
#         }

#     def solve(self, initial_prompt: str, 
#               num_thoughts: Optional[int] = None, 
#               max_steps: Optional[int] = None, 
#               max_states: Optional[int] = None, 
#               value_threshold: Optional[float] = None, 
#               pruning_threshold: Optional[float] = 0.5,
#             ):
#         start_time = time.time()
#         self.file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"
#         try:
#             best_thoughts = ""
#             if self.search_algorithm == 'BFS':
#                 result = self.tot_bfs(initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     best_thoughts = result
#             elif self.search_algorithm == 'DFS':
#                 result = self.tot_dfs(initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold)
#                 if result:
#                     self.save_tree_to_json(self.file_name)
#                     best_thoughts = result
#             if best_thoughts:
#                 solution = self.model.generate_solution(initial_prompt, best_thoughts)
#                 if solution:
#                     return solution
#             else:
#                 raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
#         except KeyboardInterrupt:
#             logger.error("Keyboard interrupt detected.")
#         except ValueError as e:
#             logger.error(f"Error: {e}")
#         finally:
#             logger.info("Saving the current tree and metrics.")
#             self.save_tree_to_json(self.file_name)

#     def logNewState(self, state, evaluation):
#         if not (type(state) == str):
#             state = " | ".join(state)
#         self.tree["nodes"][state] = evaluation
#         self.save_tree_to_json(self.file_name)    
    
#     #reject condition conditioning
#     def reject_condition(self, current_path, rejected_path, reason, value_threshold):
#         if reason["value"] < value_threshold:
#             return True
#         return False
    
#     def parse_evaluation(self, evaluation):
#         parsed_evaluation = re.findall(r'`(\d+\.\d+)`', evaluation)
#         return parsed_evaluation
    



#     #takes rejected state value reason -> injects into the the generate solutions prompt.
#     def update_current_branch(self, current_path, rejected_path_info):
#         parsed_evaluation = self.parse_evaluation(rejected_path_info["evaluation"])
#         updated_solution= f"{current_path} {parsed_evaluation}"
#         return updated_solution

#     #rejected states passing into next thoughts
#     def integrate_rejected_paths(self, current_path):
#         for rejected_path, reason in self.tree['rejected_paths'].items():
#             if self.reject_condition(current_path, rejected_path, reason):
#                 updated_solution = self.update_current_branch(current_path, reason)
#                 current_path = updated_solution

        
#     def tot_bfs(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold):
#         current_states = [initial_prompt]
#         state_values = {}
#         try:
#             for step in range(1, max_steps + 1):
#                 selected_states = []
#                 for state in current_states:
#                     # self.integrate_rejected_paths(state)
#                     #for thought in thoughts:
#                     thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
#                                                             # rejected_solutions=state)
#                     evaluated_thoughts = self.model.evaluate_states({thought: 0 for thought in thoughts}, initial_prompt)
#                     for thought, value in evaluated_thoughts.items():
#                         if value >= pruning_threshold:
#                             flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
#                             selected_states.append(flattened_state)
#                             state_values[flattened_state] = value
#                             self.logNewState(flattened_state, value)
#                         #Rejected loop pruning loop
#                         # else:
#                         #     reason_for_rejection = {"value": value, "evaluation": evaluation}
#                         #     self.tree["rejected_paths"][flattened_state] = reason_for_rejection

#                 if len(selected_states) > 1:
#                     current_states = selected_states[:max_states]

#             if len(current_states) == 1:
#                 return initial_prompt

#             if current_states:
#                 best_state = max(current_states, key=lambda state: state_values[state])
#                 return best_state
#         except Exception as e:
#             logger.error(f"Error in tot_bfs: {e}")
#             return None

#     def tot_dfs(self,
#                 initial_prompt: str,
#                 num_thoughts: str,
#                 max_steps: int,
#                 value_threshold,
#                 pruning_threshold=0.5):
#         output = []
#         file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"

#         def dfs(state, step):
#             nonlocal output
#             if step > max_steps:
#                 thought = self.model.generate_thoughts(state, 1, initial_prompt)
#                 value = self.model.evaluate_states({state}, initial_prompt)[state]
#                 output.append((thought, value))
#                 return 
            
#             for next_state in sorted(self.model.generated_thoughts(state, num_thoughts, initial_prompt)):
#                 state_value = self.model.evaluate_states({next_state}, initial_prompt)[next_state]
#                 logger.ingo(f"state: {next_state}, value: {state_value}")


#                 if state_value > value_threshold and (pruning_threshold is None or state_value >= pruning_threshold):
#                     if isinstance(state, str):
#                         child = (state, next_state)
#                     else:
#                         child = (*state, next_state)

#                     dfs(child, step + 1)

#             self.save_tree_to_json(file_name)
        
#         try:
#             dfs(initial_prompt, 1)
#             best_state = max(output, key=lambda x: x[1])
#             return best_state[0]
#         except Exception as e:
#             logger.error(f"Error in tot_dfs: {e}")
#             return None


#     def save_tree_to_json(self, file_name):
#         os.makedirs(os.path.dirname(file_name), exist_ok=True)

#         with open(file_name, 'w') as json_file:
#             json.dump(self.tree, json_file, indent=4)

#     def print_tree(self, 
#                    node: str, 
#                    depth=0):
#         thought = self.tree["metrics"]["thoughts"].get(node, "")
#         evaluation = self.tree["metrics"]["evaluations"].get(node, "")

#         tree_info = f"{'  ' * depth}Node: {node}, Thought: {thought}, Evaluation: {evaluation}\n"

#         for child, parent in self.tree["nodes"].items():
#             if parent == node:
#                 tree_info += self.print_tree(child, depth + 1)
#                 print(f'tree info: {tree_info}')

#         return tree_info
 # def monte_carlo_search(self,
    #                     initial_prompt: str,
    #                     num_thoughts: int,
    #                     max_steps: int,
    #                     max_states: int,
    #                     pruning_threshold: float,
    #                     # sleep_time: float,
    #                     ):
    #     current_states = [initial_prompt]
    #     state_values = {}
    #     for step in range(1, max_steps + 1):
    #         selected_states = []
    #         for state in current_states:
    #             thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
    #             # time.sleep(sleep_time)
    #             evaluated_thoughts = self.model.evaluate_states(thoughts, initial_prompt)
    #             self.tree["metrics"]["thoughts"].update({thought: state for thought in thoughts})
    #             self.tree["metrics"]["evaluations"].update(evaluated_thoughts)
    #             for thought, value in evaluated_thoughts.items():
    #                 if value >= pruning_threshold:
    #                     flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
    #                     selected_states.append(flattened_state)
    #                     state_values[flattened_state] = value
    #                     self.logNewState(flattened_state, value)
    #         if len(selected_states) > max_states:
    #             current_states = selected_states[:max_states]
    #         self.save_tree_to_json(self.file_name)
    #         if state_values:  # Check if state_values is not empty
    #             best_state = max(state_values, key=state_values.get)
    #             solution = self.model.generate_solution(initial_prompt, best_state)
    #             self.solution_found = True
    #             return solution
            
    #         num_thoughts, max_steps, max_states = self.optimize_params(num_thoughts, max_steps, max_states)

    #     return None  # or an indication of no solution found



#v2
    # def monte_carlo_search(self,
    #                 initial_prompt: str,
    #                 num_thoughts: int,
    #                 max_steps: int,
    #                 max_states: int,
    #                 pruning_threshold: float,
    #                 ):
    #     current_states = [initial_prompt]
    #     state_values = {}
    #     visit_counts = {}
    #     for step in range(1, max_steps + 1):
    #         selected_states = []
    #         for state in current_states:
    #             thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
    #             time.sleep(2)
    #             evaluated_thoughts = self.model.evaluate_states(thoughts, initial_prompt)
    #             print(f"evaluated thoughts: {evaluated_thoughts}")
    #             self.tree["metrics"]["thoughts"].update({thought: state for thought in thoughts})
    #             self.tree["metrics"]["evaluations"].update(evaluated_thoughts)
    #             for thought, value in evaluated_thoughts.items():
    #                 flattened_state = (state, thought) if isinstance(state, str) else (*state, thought)
    #                 visit_counts[flattened_state] = visit_counts.get(flattened_state, 1) + 1
    #                 if state in visit_counts:
    #                     if value + np.sqrt(2 * np.log(visit_counts[state]) / visit_counts[flattened_state]) >= pruning_threshold:
    #                         selected_states.append(flattened_state)
    #                         state_values[flattened_state] = value
    #                         self.logNewState(flattened_state, value)
    #         if len(selected_states) > max_states:
    #             current_states = selected_states[:max_states]
    #         self.save_tree_to_json(self.file_name)
    #         if state_values:  # Check if state_values is not empty
    #             best_state = max(state_values, key=state_values.get)
    #             solution = self.model.generate_solution(initial_prompt, best_state)
    #             self.solution_found = True
    #             return solution

    #         num_thoughts, max_steps, max_states = self.optimize_params(num_thoughts, max_steps, max_states)
            
    #     return None  # or an indication of no solution found
