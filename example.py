import os
from tree_of_thoughts.openaiModels import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughts
#


api_model= "gpt-3.5-turbo"


model = OpenAILanguageModel(api_key='', api_model=api_model)

#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

# value or vote
evaluation_strategy = "value"

tree_of_thoughts= TreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation"

num_thoughts = 2
max_steps= 3
max_states = 5
value_threshold= 0.5

#call the solve emthod with the input problem and other params

solution = tree_of_thoughts.solve(input_problem, 
    num_thoughts=num_thoughts,
    max_steps=max_states,
    max_states=5,
    value_threshold=value_threshold,
    )
    
#use the solution in your production environment
print(f"solution: {solution}")

