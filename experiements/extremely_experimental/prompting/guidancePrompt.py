
import concurrent.futures
from abc import ABC, abstractmethod
import traceback
import openai
import os
import re
import guidance
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
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base == None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = openai.ChatCompletion.create(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    response = openai.Completion.create(
                        engine=self.api_model,
                        prompt=prompt,
                        n=k,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                    )
                return response
            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        
        prompt = f"Given the current state of reasoning: '{state_text}', generate {1} coherent thoughts to continue the reasoning process:"
        prompt += self.ReAct_prompt
        if self.use_chat_api:
            new_prompt_success = False
            """
            # Try prompt and parse in a single shot to save tokens (but if we fail, we end up spending more tokens)
            new_prompt = prompt + "Thought string should be output in a format that can be parsed into python array in format [xxx,xxx,xxx]"
            response = self.openai_api_call_handler(new_prompt, 100 * k, 0.5, 1)
            text = self.openai_choice2text_handler(response.choices[0])
            re_parse = re.search(r'\[(.*?)\]', text)
            if re_parse:
                thoughts_str = re_parse.group(1)
                if thoughts_str:
                    thoughts = thoughts_str.split(',')
                    new_prompt_success = len(thoughts) == k 
                    if not new_prompt_success:
                        print(f"Fall back to multi-prompt for chat-completion due to parse fail {text}")

            """
            if not new_prompt_success:
                thoughts = []
                for _ in range(k):
                    response = self.openai_api_call_handler(prompt, 50, 0.5, k)
                    text = self.openai_choice2text_handler(response.choices[0])
                    thoughts += [text]
            
        else:
            response = self.openai_api_call_handler(prompt, 50, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, and NOTHING ELSE:"
                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            prompt = f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote, and NOTHING ELSE:"
            response = self.openai_api_call_handler(prompt, 50, 1)
            best_state_text = self.openai_choice2text_handler(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")



class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=True):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
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
    
class GuidanceLanguageModel(AbstractLanguageModel):
    def __init__(self, model, strategy="cot", evaluation_strategy="value"):
        # guidance.llms.OpenAI("gpt-4")
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass


class GuidanceOpenAILanguageModel(GuidanceLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model=""):
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base == None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')
        
        super().__init__(guidance.llms.OpenAI(self.api_model), strategy, evaluation_strategy)
        self.evaluation_strategy = evaluation_strategy

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        
        try:
                # Replace the prompt with a Guidance program
            program = guidance(f"Given the current state of reasoning: '{{state_text}}', generate {{{k}}} coherent thoughts to continue the reasoning process:{{#each (gen 'thoughts' k=k)}}\n{{this}}{{/each}}")
            thoughts_text = program(state_text=state_text, k=k)
            thoughts = thoughts_text
            
            print(f"Generated thoughts: {thoughts}")
            return thoughts
        except Exception as e:
            print(f"Error Generating thoughts {e} ")
            traceback.print_exc()
            thoughts = []
        
        return thoughts
    
    def evaluate_states(self, states):
        # ... (rest of the method remains unchanged)

        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ''.join(state)

                try:
                    #replace prompt with a guidnace program
                    program = guidance(f"Given the current state of reasoning: '{{state_text}}', evaluate its value as a float between 0 and 1, and NOTHING ELSE:{{select 'value' options=(gen 'value')}}")
                    value_text = str(program(state_text=state_text))
                    value = float(value_text)
                    print(f"Value {value}")
                except ValueError:
                    print(f"Error parsing value: {value_text}")
                    traceback.print_exc()
                    value = 0 #assign a default value if the conversionfail

                state_values[state] = value
            return state_values

        if self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            
            # Replace the prompt with a Guidance program
            program = guidance(f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote, and NOTHING ELSE:")
            best_state_text = program(state_text=state_text)
            
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")






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

    def tot_dfs(self, x, k, T, vth, pruning_threshold=0.5, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
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
    def solve(self, x, k=5, T=3, b=5, vth=0.5, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
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






api_key = "api key"
language_model = GuidanceOpenAILanguageModel(api_key)

search_algorithm = "DFS"
#init optimized tree of thoughts
tree_of_thoughts = OptimizedTreeofThoughts(language_model, search_algorithm)

#define the inital state 

# Set the input problem and parameters for the Tree of Thoughts algorithm
input_problem = "What are next generation reasoning methods for Large Language Models"
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 1.0
max_iterations = 40
convergence_threshold = 0.01
convergence_count = 5

# Run the Tree of Thoughts algorithm
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)

# Print the solution
print(f"Solution: {solution}")