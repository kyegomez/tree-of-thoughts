#give topic [What are quantum field theorem proofs respond in math notation] -> 100 questions by external model -> tree of thoughts for each question
#give dataset -> ask questions about each example and fine tune on like alpaca dataset
import json
from tree_of_thoughts.treeofthoughts import OptimizedTreeofThoughts
from tree_of_thoughts.treeofthoughts import OptimizedOpenAILanguageModel

k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 1.0 #cmodel is confident on performance
max_iterations = 40 #tree branch nodes 
convergence_threshold = 0.01
convergence_count = 5


class DatasetGenerator:
    def __init__(self, openai_language_model, tree_of_thoughts):
        self.openai_language_model = openai_language_model
        self.tree_of_thoughts = OptimizedOpenAILanguageModel(openai_api_key, api_model="gpt-3.5-turbo")

    def generate_questions(self, topic, n_questions=100):
        prompt=f"Generate {n_questions} unique questions related to the topic '{topic}':"
        response = self.openai_language_model.openai_api_call_handler(prompt, 50 * n_questions, 0.5, 1)
        questions_text = self.openai_language_model.openai_choice2text_handler(response.choices[0])
        questions = questions_text.split('\n')[:n_questions]
        return questions
    

    def generate_dataset(self, topic, n_questions: 1000):
        questions = self.generate_questions(topic, n_questions)
        dataset = []
        
        for question in questions:
            # solution = self.tree_of_thought.solve(question)
            solution = tree_of_thoughts.solve(question, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
            dataset_entry = {
                "question": question,
                "solution": solution
            }
            dataset.append(dataset_entry)

        return dataset
    


openai_api_key=""

# openai_language_model = OptimizedOpenAILanguageModel(openai_api_key, api_model="gpt-3.5-turbo")
tree_of_thoughts = OptimizedTreeofThoughts(search_algorithm="DFS")

dataset_generator = DatasetGenerator(tree_of_thoughts)
topic = "Artificial Intelligence"
dataset = dataset_generator.generate_dataset(topic)

# Save the dataset to a JSON file
with open("tot_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
















# def generate_dataset(self, topic, n_questions=100):
#     questions = self.generate_questions(topic, n_questions)
#     dataset = []

#     for question in questions:
#         thoughts_and_evals = []
#         state = [question]
#         solution_found = False
#         while not solution_found:
#             thoughts = self.guidance_language_model.generate_thoughts(state, k)
#             state_values = self.guidance_language_model.evaluate_state(thoughts)
#             best_thought = self.select_best_thought(thoughts, state_values[best_thought])
#             thoughts_and_evals.append((best_thought, state_values[best_thought]))
#             state.append(best_thought)
#             if self.is_solution(best_thought):
#                 solution_found = True
        
#         dataset_entry = {
#             "question": question,
#             "instructions": thoughts_and_evals,
#             "solution": best_thought
#         }
#         dataset.append(dataset_entry)

#     return dataset

# def is_solution(self, thought):
#     #implement the logic to determine if the thought is a solution
#     pass