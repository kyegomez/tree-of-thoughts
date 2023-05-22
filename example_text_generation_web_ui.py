from tree_of_thoughts import TextGenerationWebUILanguageModel, TreeofThoughts, OptimizedTreeofThoughts

#v1
model = TextGenerationWebUILanguageModel()

#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

#create an instance of the tree of thoughts class v1
tree_of_thoughts = TreeofThoughts(model, search_algorithm)

#or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts
tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "What are the next generation reasoning methods for Large Language Models"
k = 5
T = 3
b = 5
vth = 0.5

# # Optimal nominal values for the stopping conditions

# confidence = 0.9 #HIGH QUALITY SOLIUTION FOUND

# max_iterations = 5 # MAX ITERATIONS 10

# convergence_threshold = 0.01 #Convergence Check: Monitor the change in evaluation values between consecutive iterations. If the change in evaluation values is below a certain threshold for a specified number of consecutive iterations, the algorithm can stop and return the solution.

# convergence_count = 5

#call the solve method with the input problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth)

#use the solution in env
print(f"solution: {solution}")
