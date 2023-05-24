from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, GuidanceOpenAILanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts

use_v2 = False
use_guidance = False
api_key="enter api key"
api_base= "" # leave it blank if you simply use default openai api url
api_model= "gpt-4"

if not use_v2:
    if not use_guidance:
        #v1
        model = OpenAILanguageModel(api_key=api_key, api_base=api_base, api_model=api_model)
    else:
        #v1 with guidance
        model = GuidanceOpenAILanguageModel(api_key=api_key, api_base=api_base, api_model=api_model)
else:
    #v2 parallel execution, caching, adaptive temperature
    model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base)

#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

if not use_v2:
    #create an instance of the tree of thoughts class v1
    tree_of_thoughts = TreeofThoughts(model, search_algorithm)
else:
    #or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts
    tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 1.0 #cmodel is confident on performance
max_iterations = 40 #tree branh nodes 
convergence_threshold = 0.01
convergence_count = 5



    
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
    
#use the solution in your production environment
print(f"solution: {solution}")
