from tree_of_thoughts.models.openai_models import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughtsDFS

#


api_model = "gpt-3.5-turbo"


model = OpenAILanguageModel(api_key="api key", api_model=api_model)

# choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

# value or vote
evaluation_strategy = "value"

tree_of_thoughts = TreeofThoughtsDFS(model)  # search_algorithm)

# Note to reproduce the same results from the tree of thoughts paper if not better,
# craft an 1 shot chain of thought prompt for your task below
input_problem = """


Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
Possible next steps:


"""

num_thoughts = 1
max_steps = 3
max_states = 3
value_threshold = 0.5

# call the solve emthod with the input problem and other params

solution = tree_of_thoughts.solve(
    input_problem,
    # num_thoughts=num_thoughts,
    max_steps=max_states,
    # max_states=max_states,
    value_threshold=value_threshold,
)

# use the solution in your production environment
print(f"solution: {solution}")
