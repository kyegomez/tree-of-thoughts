# Comprehensive Documentation and Changelog
This document provides a comprehensive overview of the changes made to the TreeofThoughts class and its methods to improve readability and understandability. The changes include updating variable names to be more meaningful and descriptive, as well as modifying the structure of the code for better readability.

## Changelog
1. TreeofThoughts Class
Updated the class definition to include a more descriptive docstring.
2. __init__ Method
No changes were made to the __init__ method.
3. solve Method
Updated variable names:
x -> initial_prompt
k -> num_thoughts
T -> max_steps
b -> max_states
vth -> value_threshold
4. tot_bfs Method
Updated variable names:
x -> initial_prompt
k -> num_thoughts
T -> max_steps
b -> max_states
S0 -> current_states
S0_t -> generated_states
Vt -> state_values
St -> selected_states
5. tot_dfs Method
Updated variable names:

x -> initial_prompt
k -> num_thoughts
T -> max_steps
vth -> value_threshold
s -> state
t -> step
s_prime -> next_state
child -> child_state

### Added optional parameters for better control over the search process:
pruning_threshold
confidence_threshold
max_iterations
convergence_threshold
convergence_count
6. save_tree_to_json Method
No changes were made to the save_tree_to_json method.
7. print_tree Method
No changes were made to the print_tree method.

# Documentation
TreeofThoughts Class
The TreeofThoughts class is designed to solve problems using a tree-based search algorithm. It takes a model and a search algorithm (either 'BFS' or 'DFS') as input and provides methods to solve problems using the chosen algorithm.

## Initialization
The __init__ method initializes the TreeofThoughts class with the given model and search algorithm. It also initializes an empty tree structure to store the search results.

## Solve Method
The solve method is the main entry point for solving problems using the TreeofThoughts class. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.
max_states: The maximum number of states to consider at each step (for BFS).
value_threshold: The threshold value for pruning states (for DFS).
timeout: The maximum time allowed for the search process.
confidence_threshold: The confidence threshold for stopping the search.
max_iterations: The maximum number of iterations allowed for the search.
convergence_threshold: The threshold for determining convergence.
convergence_count: The number of consecutive convergences required to stop the search.
Based on the chosen search algorithm, the solve method calls either the tot_bfs or tot_dfs method to perform the search.

## tot_bfs Method
The tot_bfs method performs a breadth-first search to solve the problem. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.
max_states: The maximum number of states to consider at each step.
pruning_threshold: The threshold value for pruning states.
The method generates and evaluates states at each step, selecting the best states based on their values. The search continues until the maximum number of steps is reached, and the best state is returned.

## tot_dfs Method
The tot_dfs method performs a depth-first search to solve the problem. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.

value_threshold: The threshold value for pruning states.
pruning_threshold: The threshold value for pruning states based on their values.
confidence_threshold: The confidence threshold for stopping the search.
max_iterations: The maximum number of iterations allowed for the search.
convergence_threshold: The threshold for determining convergence.
convergence_count: The number of consecutive convergences required to stop the search.
The method uses a recursive depth-first search approach to explore the state space. It generates and evaluates states at each step, and if a state's value is above the value_threshold and pruning_threshold, it continues the search with the new state. The search stops when the maximum number of steps is reached, the confidence threshold is met, or the convergence criteria are satisfied. The best state is then returned.

## save_tree_to_json Method
The save_tree_to_json method saves the current tree structure and metrics to a JSON file. It takes the following parameter:

file_name: The name of the JSON file to save the tree structure and metrics.
This method is useful for logging the search process and analyzing the results later.

## print_tree Method
The print_tree method prints the tree structure in a human-readable format. It takes the following parameters:

node: The current node in the tree.
depth: The depth of the current node in the tree (default is 0).
This method is useful for visualizing the tree structure and understanding the search process.

## Usage
To use the TreeofThoughts class, follow these steps:

Initialize the class with a model and a search algorithm (either 'BFS' or 'DFS').
Call the solve method with the required parameters to perform the search and obtain the best state.
(Optional) Use the save_tree_to_json method to save the tree structure and metrics to a JSON file.
(Optional) Use the print_tree method to visualize the tree structure.
Here's an example of how to use the TreeofThoughts class: