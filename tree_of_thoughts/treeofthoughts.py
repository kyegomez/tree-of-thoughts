
import concurrent.futures
from abc import ABC, abstractmethod
import openai
import os
import re
import guidance
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import heapq
import json
DATA_PATH = './data'
import logging 
import argparse 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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



class HuggingLanguageModel(AbstractLanguageModel):
    def __init__(self, model_name, model_tokenizer=None, verbose=False):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer or model_name)
        self.verbose = verbose

    def generate_thoughts(self, state, k, max_length=100):
        state_text = ' '.join(state)
        prompt = f"Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx Given the current state of reasoning: '{state_text}', generate {k} coherent solutions to achieve {state_text}"

        if self.verbose:
            print(f"Generating thoughts for state: {state_text}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=k)
            thoughts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            if self.verbose:
                print(f"Error generating thoughts for state: {state_text}")
                print(f"Error: {e}")
            thoughts = []

        return thoughts

    def evaluate_states(self, states, inital_prompt, max_length=10):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the current state of reasoning: '{state_text}', pessimitically evaluate its value as a float between 0 and 1 based on it's potential to achieve {inital_prompt}"

            if self.verbose:
                print(f"Evaluating state: {state_text}")

            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs, num_return_sequences=1, max_length=max_length)
                value_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                value = float(value_text)
            except ValueError:
                if self.verbose:
                    print(f"Error converting value to float for state: {state_text}")
                value = 0  # Assign a default value if the conversion fails
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating state: {state_text}")
                    print(f"Error: {e}")
                value = 0

            state_values[state] = value

        return state_values

@staticmethod
class HFPipelineModel(AbstractLanguageModel):
    def __init__(self, model_name, verbose=False):
        self.model_name = model_name
        self.pipeline = pipeline("text-generation", model=model_name)
        self.verbose = verbose

    def generate_thoughts(self, state, k, max_length=100):
        state_text = ' '.join(state)
        prompt = f"Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx Given the current state of reasoning: '{state_text}', generate {k} coherent solutions to achieve"

        if self.verbose:
            print(f"Generating thoughts for state: {state_text}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs["input_ids"], max_length=max_length, num_return_sequences=k)
            thoughts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            if self.verbose:
                print(f"Error generating thoughts for state: {state_text}")
                print(f"Error: {e}")
            thoughts = []

        return thoughts

    def evaluate_states(self, states, initial_prompt, max_length=10):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the current state of reasoning: '{state_text}', pessimistically evaluate its value as a float between 0 and 1 based on its potential to achieve {initial_prompt}"

            if self.verbose:
                print(f"Evaluating state: {state_text}")

            try:
                generated_outputs = self.pipeline(prompt, max_length=max_length, num_return_sequences=1)
                value_text = generated_outputs[0]["generated_text"]
                value = float(value_text)
                print(f'value {value}')
            except ValueError:
                if self.verbose:
                    print(f"Error converting value to float for state: {state_text}")
                value = 0  # Assign a default value if the conversion fails
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating state: {state_text}")
                    print(f"Error: {e}")
                value = 0

            state_values[state] = value

        return state_values
    
    @staticmethod
    def load(model_nmae, verbose=False):
        return HFPipelineModel(model_name, verbose)
    
        




class OpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        env_tree = os.env("OPENAI_API_KEY")
        if api_key == "" or api_key or env_tree == None:
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
                with open("openai.logs", 'a') as log_file:
                    log_file.write("\n" + "-----------" + '\n' +"Prompt : "+ prompt+"\n")
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
    
    def generate_text(self, prompt, k):
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = self.openai_api_call_handler(prompt, 50, 0.5, k)
                text = self.openai_choice2text_handler(response.choices[0])
                thoughts += [text]
                print(f'thoughts: {thoughts}')
            return thoughts
            
        else:
            response = self.openai_api_call_handler(prompt, 50, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
            return thoughts


    # def generate_thoughts(self, state, k):
    #     state_text = ' '.join(state)
        
    #     prompt = f"Given the current state of objective: '{state_text}', generate the next coherent {k} solutions to achieve the objective: "
    #     prompt += self.ReAct_prompt
    #     print(prompt)
    #     if self.use_chat_api:
    #         new_prompt_success = False
    #         """
    #         # Try prompt and parse in a single shot to save tokens (but if we fail, we end up spending more tokens)
    #         new_prompt = prompt + "Thought string should be output in a format that can be parsed into python array in format [xxx,xxx,xxx]"
    #         response = self.openai_api_call_handler(new_prompt, 100 * k, 0.5, 1)
    #         text = self.openai_choice2text_handler(response.choices[0])
    #         re_parse = re.search(r'\[(.*?)\]', text)
    #         if re_parse:
    #             thoughts_str = re_parse.group(1)
    #             if thoughts_str:
    #                 thoughts = thoughts_str.split(',')
    #                 new_prompt_success = len(thoughts) == k 
    #                 if not new_prompt_success:
    #                     print(f"Fall back to multi-prompt for chat-completion due to parse fail {text}")

    #         """
    #         if not new_prompt_success:
    #             thoughts = []
    #             for _ in range(k):
    #                 response = self.openai_api_call_handler(prompt, 50, 0.5, k)
    #                 text = self.openai_choice2text_handler(response.choices[0])
    #                 thoughts += [text]
    #                 print(f'thoughts: {thoughts}')
            
    #     else:
    #         response = self.openai_api_call_handler(prompt, 50, 0.5, k)
    #         thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
    #     # print(thoughts)
    #     print(f"Generated thoughts: {thoughts}")
    #     return thoughts

    def generate_thoughts(self, state, k):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("We receive a state of type", type(state), "For state: ", state, "\n\n")
        
        prompt = f"Given the current state of reasoning: \n\n\n'{state_text}'\n\n\nGenerate the next best coherent thought to achieve the reasoning process and get the solution: "
        prompt += self.ReAct_prompt
        print(prompt)
        thoughts = self.generate_text(prompt, k)
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

        
    def generate_solution(self, initial_prompt, state):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        
        prompt = f"Given the following reasoning: \n\n\n'{state_text}'\n Give me the best solution you can think of the task : {initial_prompt}"
        answer = self.generate_text(prompt, 1)
        # print(thoughts)
        print(f"General solution : {answer}")
        return answer

    def evaluate_states(self, states, inital_prompt):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {inital_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"
                
                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {inital_prompt} and become very pessimistic very NOTHING ELSE"

            response = self.openai_api_call_handler(prompt, 50, 1)

            print(f'state response: {response}')

            best_state_text = self.openai_choice2text_handler(response.choices[0])

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f'best_state: {best_state}')

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
    # def solution(self, states, initial_prompt):

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=False):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states, inital_prompt):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states, inital_prompt))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    
class GuidanceLanguageModel(AbstractLanguageModel):
    def __init__(self, model, strategy="cot", evaluation_strategy="value", enable_ReAct_prompting=False):
        # gpt4 = guidance.llms.OpenAI("gpt-4")
        # vicuna = guidance.llms.transformers.Vicuna("your_path/vicuna_13B", device_map="auto")
        self.model = model
        
        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = '''{{#assistant~}}
            {{gen 'Observation' temperature=0.5 max_tokens=50}}
            {{~/assistant}}'''
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        
        self.thoughts_program = guidance('''
            {{#system~}}
            You are a logical and rational assistant.
            {{~/system}}

            {{#user~}}
            Given the current state of reasoning:
            {{state_text}}
            Generate {{k}} coherent thoughts as short as possible to continue the reasoning process.
            Don't answer the question yet.
            {{~/user}}

            %s
            
            {{#assistant~}}
            {{gen 'Thoughts' temperature=0.5 max_tokens=50}}
            {{~/assistant}}
            ''' % self.ReAct_prompt, llm=self.model)
        
        self.value_program = guidance('''
            {{#system~}}
            You are a logical and rational assistant.
            {{~/system}}

            {{#user~}}
            Given the current state of reasoning:
            {{state_text}}
            Evaluate its value as a float between 0 and 1, and NOTHING ELSE
            Don't answer the question yet.
            {{~/user}}

            {{#assistant~}}
            {{gen 'Value' temperature=1 max_tokens=10}}
            {{~/assistant}}
            ''', llm=self.model)
        
        self.vote_program = guidance('''
            {{#system~}}
            You are a logical and rational assistant.
            {{~/system}}

            {{#user~}}
            Given the following states of reasoning, vote for the best state:
            {{states_text}}
            Give the index of your voted best state(the 1st state has index 0), and NOTHING ELSE
            Don't answer the question yet.
            {{~/user}}

            {{#assistant~}}
            {{gen 'Vote' temperature=1 max_tokens=10}}
            {{~/assistant}}
            ''', llm=self.model)
        
    def model_response_handler(self, program, **kargs):
        print("Calling guidance model(Modify Me to handle specific LLM response excpetions!)")
        reponse = program(**kargs)
        return reponse

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        state_text = ' '.join(state)
        
        thoughts = []
        for _ in range(k):
            response = self.model_response_handler(self.thoughts_program, state_text=state_text, k=1)
            text = response['Thoughts']
            thoughts += [text]
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                response = self.model_response_handler(self.value_program, state_text=state_text)
                try:
                    value_text = response['Value']
                    print(f"Value text {value_text}")
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            response = self.model_response_handler(self.vote_program, states_text=states_text)
            best_state_text = response['Vote']
            print(f"Best state text: {best_state_text}")
            best_state = int(best_state_text)
            return {state: 1 if i == best_state else 0 for i in range(len(states))}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class GuidanceOpenAILanguageModel(GuidanceLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=False):
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

        super().__init__(guidance.llms.OpenAI(self.api_model), strategy, evaluation_strategy, enable_ReAct_prompting)
        
    
    def model_response_handler(self, program, **kargs):
        error_msg = ''
        while True:
            try:
                program.llm.max_retries = 60
                guidance.llms.OpenAI.cache.clear()
                response = program(**kargs)
                return response
            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)
            except Exception as e:
                if str(e) == f'''Too many (more than {guidance.llm.max_retries}) OpenAI API RateLimitError's in a row!''':
                    sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                    print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                    time.sleep(sleep_duratoin)
                else:
                    error_msg = str(e)
                    break
        raise Exception(error_msg)


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
        self.tree = {
            "nodes": [],
            "metrics": {
                "thoughts": [],
                "evaluations": []
            }
        }

    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"
        try:
            if self.search_algorithm == 'BFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_bfs(x, k, T, b)
                    if result:
                        self.save_tree_to_json(file_name)
                        return result
            elif self.search_algorithm == 'DFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_dfs(x, k, T, vth)
                    if result:
                        self.save_tree_to_json(file_name)
                        return result
            else:
                raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
        except KeyboardInterrupt:
            logger.error("Keyboard interrupt detected.")
        except ValueError as e:
            logger.error(f"Error: {e}")
        finally:
            logger.info("Saving the current tree and metrics.")
            self.save_tree_to_json(file_name)



    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = set()
            for s in S0:
                for z in self.model.generate_thoughts(s, k):
                    if (type(s) == str):
                        S0_t.add((s, z))
                    else:
                        S0_t.add((*s, z))
            Vt = self.model.evaluate_states(S0_t, x)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)

            logger.info(f'Step: {t}, S0_t: {S0_t}, Vt: {Vt}, St: {St}, S0: {S0}')



        best_state = max(St, key=lambda s: Vt[s])

        return best_state


    def tot_dfs(self, x, k, T, vth, pruning_threshold=0.5, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        output = []
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None
        file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"


        def dfs(s, t):
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count, output
            if t > T:
                thought = self.model.generate_thoughts(s, 1)
                print(f'thoughts inside dfs {thought}')
                
                value = self.model.evaluate_states({s}, x)[s]
                print(f'values inside dfs {value}')

                output.append((thought, value))
                print(f'output {output}')

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
                state_value = self.model.evaluate_states({s_prime}, x)[s_prime]
                logger.info(f"State: {s_prime}, Value: {state_value}")

                if state_value > vth and (pruning_threshold is None or state_value >= pruning_threshold):
                    if (type(s) == str):
                        child = (s, s_prime)
                    else:
                        child = (*s, s_prime)
                    # self.tree['nodes'][child] = s
                    # self.tree["metrics"]["thoughts"][child] = s_prime
                    # self.tree["metrics"]["evaluations"][child] = state_value

                    if dfs(child, t + 1):
                        return True

            self.save_tree_to_json(file_name)
            return False
        
            
        dfs(x, 4)
        print(f'output  {output}')
        best_state = max(output, key=lambda x: x[1])
        return best_state[0]

    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def print_tree(self, x, node=None, depth=0):
        if node is None:
            node = self.tree["nodes"][x]

        thought = self.tree["metrics"]["thoughts"][node]
        evaluation = self.tree["metrics"]["evaluations"][node]

        tree_info = {
            "node": node,
            "thought": thought,
            "evaluation": evaluation,
            "children": []
        }

        for child, parent in self.tree["nodes"].items():
            if parent == node:
                child_info = self.print_tree(child, depth + 1)
                tree_info["children"].append(child_info)

        return tree_info


#does not output state after each thought --- idk why -- needs work
class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        print(f'Start time {start_time}')
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                print(f'resultttt in optimized tree of thoughts: {result}')
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth, confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")



if __name__ == '__main__':
    search_algorithm = "DFS"
    strategy = "cot"
    evaluation_strategy="vote"
    
    #create instance
    model = OptimizedOpenAILanguageModel('', api_model="gpt-3.5-turbo")
    


    tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm)

    # input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

    parser = argparse.ArgumentParser(description="Tree of Thoughts Solver")

    parser.add_argument("--problem", type=str, required=True, help="Initial problem statement")
    parser.add_argument("--search_algorithm", type=str, choices=["BFS", "DFS"], default="BFS", help="Search algorithm to use (BFS or DFS)")
    parser.add_argument("--k", type=int, default=3, help="Number of thoughts to generate")
    parser.add_argument("--T", type=int, default=10, help="Step limit")
    parser.add_argument("--b", type=int, default=5, help="Number of most promising states")
    parser.add_argument("--vth", type=float, default=0.4, help="Value threshold for DFS")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds before stopping")
    parser.add_argument("--confidence", type=float, default=0.8, help="Model confidence threshold")
    parser.add_argument("--max_iterations", type=int, default=40, help="Maximum number of tree branch nodes")
    parser.add_argument("--convergence_threshold", type=float, default=0.01, help="Convergence threshold for the search process")
    parser.add_argument("--convergence_count", type=int, default=5, help="Number of searches to be considered converged")

    args = parser.parse_args()

    #solve the problem using the tree of thoughts class
    optimized_tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm=args.search_algorithm)

    #solve the porblem using tree of thoughts problem helper
    best_state = optimized_tree_of_thoughts.solve(args.problem, k=args.k, T=args.T, b=args.b, vth=args.vth)


    #generate the final silution
    final_solution = optimized_tree_of_thoughts.model.generate_solution(best_state, args.problem)


    #print the final solutions
    print(f"Final solution: {final_solution}")

    trees = optimized_tree_of_thoughts.print_tree(final_solution)


    # tree_of_thoughts.print_tree(final_solution)


    #generate solution prompt --> give me an solution is right now -> get_final_answer that takes into account the best state and provides the response
    #python tree_of_thoughts.py --problem "Design a new transportation system for a city" --search_algorithm BFS --k 5 --T 3 --b 5 --vth 0.5 --timeout 10 --confidence 0.8 --max_iterations 40 --convergence_threshold 0.01 --convergence_count 5

    
