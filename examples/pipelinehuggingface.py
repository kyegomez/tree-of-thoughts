from tree_of_thoughts.treeofthoughts import HFPipelineModel, OptimizedTreeofThoughts
from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, GuidanceOpenAILanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts, HuggingLanguageModel


model_name="gpt2"
gpt2_pipeline_model = HFPipelineModel(model_name)

tree_of_thoughts = TreeofThoughts(gpt2_pipeline_model, search_algorithm="DFS")
#



#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"



#input your own objective if you will
input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

#play around for increase in performance
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

