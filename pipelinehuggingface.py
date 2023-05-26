from tree_of_thoughts.treeofthoughts import HFPipelineModel, OptimizedTreeofThoughts

model_name="gpt2"
gpt2_pipeline_model = HFPipelineModel(model_name)

tree_of_thoughts = OptimizedTreeofThoughts(gpt2_pipeline_model, search_algorithm="dfs")


from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, GuidanceOpenAILanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts, HuggingLanguageModel
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # Initialize the HuggingLanguageModel with the GPT-2 model
# model_name = "gpt2"
# model = HuggingLanguageModel(model_name, 
#                              model_Tokenizer="gpt2", 
#                              verbose=True)
                             



#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"


# gpt2_model = HuggingLanguageModel(model_name)

# tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 0.8 #cmodel is confident on performance
max_iterations = 40 #tree branh nodes 
convergence_threshold = 0.01
convergence_count = 5

    
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
    
#use the solution in your production environment
print(f"solution: {solution}")
