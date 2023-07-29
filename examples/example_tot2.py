from tree_of_thoughts.openai_models import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughts2
#


api_model= "gpt-3.5-turbo"


model = OpenAILanguageModel(api_key='api key', api_model=api_model)



tree_of_thoughts= TreeofThoughts2(model) #search_algorithm)

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

# Solve a problem with the TreeofThoughts

num_thoughts = 1
max_steps = 2
pruning_threshold = 0.5

solution = tree_of_thoughts.solve(input_problem, num_thoughts, max_steps, pruning_threshold)

print(f"solution: {solution}")

