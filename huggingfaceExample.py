from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, GuidanceOpenAILanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts, HuggingLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name="gpt2"
# model_tokenizer="gpt2tokenizer"
# model = HuggingLanguageModel(model_name, model_tokenizer)


class HuggingLanguageModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Initialize the HuggingLanguageModel with the GPT-2 model
model_name = "gpt2"
model = HuggingLanguageModel(model_name)


#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

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

gpt2_model = HuggingLanguageModel(model_name)

generated_text = gpt2_model.generate_thoughts(input_problem)
    
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
    
#use the solution in your production environment
print(f"solution: {solution}")
