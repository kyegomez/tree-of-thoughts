import concurrent.futures
from abc import ABC, abstractmethod
import openai
import time


class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass
class OpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value"):
        openai.api_key = api_key
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        
        prompt = f"Given the current state of reasoning: '{state_text}', generate {k} coherent thoughts to continue the reasoning process:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            n=k,
            max_tokens=50,
            stop=None,
            temperature=0.5,
        )
        thoughts = [choice.text.strip() for choice in response.choices]
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1:"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    n=1,
                    max_tokens=10,
                    stop=None,
                    temperature=1,
                )
                try:
                    value_text = response.choices[0].text.strip()
                    print(f"Value text {value_text}")
                    value = float(response.choices[0].text.strip())
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            prompt = f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                n=1,
                max_tokens=50,
                stop=None,
                temperature=1,
            )
            best_state_text = response.choices[0].text.strip()
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True):
        super().__init__(api_key, strategy, evaluation_strategy)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    

# model = OptimizedOpenAILanguageModel('your_openai_api_key_here')
# model = OptimizedOpenAILanguageModel('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')

#update tree of thoughts to use optimized models mehtods

class TreeofThoughts:
    """
    1. Thought Decomposition --> based on problem properties

    2. Thought Generator -> create a thought generator function G(p0, s, k) with 2 strategies a sample iid thoughts from a cot prompt b. propose thoughts
    sequentially using a propose prompt

    3. create a state evaluator function V(p0, S) with 2 strategies a value each state independently b. vote across states

    4. Choose a search algo based on tree structure [BFS or DFS]

    Implement chosen search algorithm for bfs (algo1):
        init S0 with the input x
        for t = 1 to T (step limit):
            generate candidate thoughts for each state in St-1
            eveluate the candiate states using the state evaluator V
            select the b most promising states for St

        return the final output by genertaing the thought for the best state in St for DFS(algo2)

        defien a recurseive DFS function with the current state s, step t, and other required params

        if t > T record the output by generating the thought for current state S

        for each candidate state s in the sorted list of generated thoughts for s:
            
            if the evaluated value of s is greater the the threshold of vth call the dfs function recursively
            with s and t + 1

    execute the chosen search algo with the input problem, thought generator, and state evaluator, and other required params
    """

    def __init__(self, model, search_algorithm):
        self.model = model
        self.search_algorithm = search_algorithm

    def solve(self, x, k, T, b, vth, timeout=None):
        start_time = time.time()
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")

    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = {(*s, z) for s in S0 for z in self.model.generate_thoughts(s, k)}
            Vt = self.model.evaluate_states(S0_t)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)
        return self.model.generate_thoughts(max(St, key=lambda s: Vt[s]), 1)

    def tot_dfs(self, x, k, T, vth, pruning_threshold=0.5, confidence_threshold=0.9, max_iterations=10, convergence_threshold=0.1, convergence_count=5):
        output = []
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None

        def dfs(s, t):
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count
            if t > T:
                thought = self.model.generate_thoughts(s, 1)
                value = self.model.evaluate_states({s})[s]
                output.append((thought, value))

                if confidence_threshold is not None and value >= confidence_threshold:
                    return True

                if prev_best_value is not None and convergence_threshold is not None:
                    if abs(value - prev_best_value) < convergence_threshold:
                        consecutive_convergence_count += 1
                    else:
                        consecutive_convergence_count = 0

                prev_best_value = value
                iteration_count += 1

                if (max_iterations is not None and iteration_count >= max_iterations) or (convergence_count is not None and consecutive_convergence_count >= convergence_count):
                    return True

                return False

            for s_prime in sorted(self.model.generate_thoughts(s, k)):
                state_value = self.model.evaluate_states({s_prime})[s_prime]
                if state_value > vth and (pruning_threshold is None or state_value >= pruning_threshold):
                    if dfs((*s, s_prime), t + 1):
                        return True

            return False

        dfs(x, 1)
        return max(output, key=lambda x: x[1]) if output else None


class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k, T, b, vth, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth, confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")

search_algorithm = "DFS"
strategy = "cot"
evaluation_strategy="vote"

#create instance
model = OptimizedOpenAILanguageModel('api-key')


tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "What is 3032 * 322 - 1"
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10

# Optimal nominal values for the stopping conditions
confidence = 0.9 #HIGH QUALITY SOLIUTION FOUND
max_iterations = 5 # MAX ITERATIONS 10
convergence_threshold = 0.01 #Convergence Check: Monitor the change in evaluation values between consecutive iterations. If the change in evaluation values is below a certain threshold for a specified number of consecutive iterations, the algorithm can stop and return the solution.
convergence_count = 5

#call the solve method with the input problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth=vth, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)

#use the solution in env
print(f"solution: {solution}")

"""
should return something like this:

['1. Utilizing reinforcement learning techniques to train large language models can be an effective approach to advancing them.\n2. Developing methods to better incorporate contextual information into large language models can help in their advancement.\n3. Incorpor', '1. Utilizing reinforcement learning techniques to allow for more efficient training of large language models.\n2. Incorporating transfer learning to leverage existing language models for faster and more accurate inference.\n3. Exploring the use of distributed', '1. Identifying and understanding key components of large language models such as natural language processing and machine learning algorithms.\n2. Utilizing methods such as transfer learning to quickly and efficiently train large language models.\n3. Incorporating', '1. Utilizing reinforcement learning techniques to train large language models can be an effective method of advancing them.\n2. Incorporating techniques such as transfer learning and data augmentation can help improve the performance of large language models.', '1. Identifying and understanding the underlying structure of language is essential to advancing large language models.\n2. Developing methods to effectively capture and represent the complexities of language is necessary for the advancement of large language models.\n3. Ut']
0.8
0.8
['4. Analyzing and interpreting large language models to identify areas of improvement.\n5. Utilizing reinforcement learning to enable models to learn from mistakes and further improve accuracy.\n6. Leveraging automated data augmentation techniques to further improve', '4. Experimenting with different architectures and hyperparameters to determine the best model for a given task.\n5. Incorporating techniques such as data augmentation and ensembling to improve the performance of large language models.\n6', '4. Exploring methods to improve the efficiency of large language models such as using distributed computing techniques.\n5. Developing methods to reduce overfitting and improve generalization of large language models.\n6. Incorporating techniques such as', '4. Exploring and utilizing different types of data sets to train large language models.\n5. Developing strategies to optimize the training process and improve the performance of large language models.\n6. Applying advanced techniques such as deep learning', '4. Exploring methods such as reinforcement learning to improve the accuracy and robustness of large language models.\n5. Utilizing data augmentation techniques to increase the amount of training data available to the model.\n6. Incorpor']
0.8
0.8
['7. Developing automated testing frameworks to validate the accuracy of large language models.\n8. Exploring ways to improve the scalability of large language models.\n9. Exploring ways to improve the efficiency of large language models.', '7. Applying methods such as active learning to further refine large language models.\n8. Developing and utilizing techniques such as knowledge distillation to compress large language models.\n9. Incorporating techniques such as semi-supervised', '7. Applying regularization techniques to reduce overfitting and improve generalization of large language models.\n8. Exploring the use of generative adversarial networks to improve the accuracy of large language models.\n9. Applying deep', '7. Developing methods to evaluate the performance of large language models on various tasks.\n8. Applying techniques such as hyperparameter tuning to optimize the performance of large language models.\n9. Utilizing adversarial training to', '7. Developing strategies to ensure large language models are able to generalize to unseen data.\n8. Incorporating methods such as meta-learning to further improve model performance.\n9. Utilizing techniques such as unsuper']
0.7
0.7
['Once the key components of large language models have been identified and understood, the best reasoning methods to advance them include utilizing transfer learning to quickly train them, analyzing and interpreting them to identify areas of improvement, leveraging reinforcement learning to enable them to learn']
0.7
0.7
['Exploring the use of meta-learning to enable models to rapidly adapt to new data and improve accuracy.']
0.7
0.7
['One potential way to further advance large language models is to incorporate automated data augmentation techniques to create more varied datasets to train the models on, as well as leveraging reinforcement learning to enable the models to learn from mistakes and continually improve accuracy.']
0.7
0.7
['By utilizing these methods, we can continue to advance large language models by improving their accuracy and performance. We can also use these methods to identify weaknesses in the models and make modifications to address them. Additionally, these methods can help us to develop']
0.7
0.7


"""