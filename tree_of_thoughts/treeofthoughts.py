
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
from functools import partial
import numpy as np
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = './data'

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class Task:
    def __init__(self):
        pass
    
    def __len__(self) -> int:
        pass

    def get_input(self, idx:int) -> str:
        pass

    def test_output(self, idx:int, output: str):
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


###########------------------------------------------< original implementation >--------------------

class TextTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """

    # standard_prompt = "Given the input text:\n\n{input}\n\nGenerate a coherent passage:"
    # cot_prompt = "Considering the input text:\n\n{input}\n\nDevise a coherent passage:"
    # vote_prompt = "Given the following passages\n\n {input}\n\n, vote for the most coherent one:\n\n"
    # value_prompt = "Considering the passage:\n\n{passage}\n\nEvaluate its coherence as a float between 0 and 1:"

    standard_prompt = '''
        Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}'''

    cot_prompt = '''
    Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

    Make a plan then write. Your output should be of the following format:

    Plan:
    Your plan here.

    Passage:
    Your passage here.
    '''


    vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.'''

    compare_prompt = '''Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".'''

    score_prompt = '''Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.'''

    # Other methods and
    def __init__(self, input_text=None):
        super().__init__()
        self.data = [input_text] if input_text else []
        self.steps = 2
        self.stops = ['\nPassage:\n', None]



    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def test_output(self, idx: int, output: str):
        output = output.split('Passage:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*coherency score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        #standard prompt
        return TextTask.standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        #prompts
        return TextTask.cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = TextTask.vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = TextTask.compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
        

###########------------------------------------------< original implementation >--------------------







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


############################################-----------------------------------------origiinal implementation
import backoff 

completion_tokens = prompt_tokens = 0

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (completion_tokens + prompt_tokens) / 1000 * 0.0002
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


############################################-----------------------------------------origiinal implementation





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
        env_tree = os.getenv("OPENAI_API_KEY")
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



    def generate_thoughts(self, state, k, inital_prompt):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("We receive a state of type", type(state), "For state: ", state, "\n\n")
        
        # prompt = f"Given the current state of reasoning: \n\n\n'{state_text}'\n\n\nGenerate the next best coherent thought to achieve the reasoning process and get the solution: "
        # prompt = f"Based on the current state of reasoning: \n\n\n'{state_text} Provide the next coherent thought that will help progress the reasoning process and reach an soluton "
        # prompt = f"These are the thoughts you've had: \n\n\n{state_text}, provide the next coherent thought that will help advance the reasoning process and reach an solution for this problem {inital_prompt}. Think sharply, think out of the box, predict failure. Do not leave any open questions. Unleash your mind."
        prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {inital_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."

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
        
        prompt = f"Considering the reasoning provided:\n\n'{state_text}'\n\nDevise the best possible solution for the task: {initial_prompt}"
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
                for z in self.model.generate_thoughts(s, k, x):
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
        #vote across across states
        output = []
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None
        file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"


        def dfs(s, t):
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count, output
            if t > T:
                thought = self.model.generate_thoughts(s, 1, x)
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

            for s_prime in sorted(self.model.generate_thoughts(s, k, x)):
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


class TreeofThoughtsv1:
    def __init__(self, model):
        self.model = model

    
    def get_value(self, task, x, y, n_evaluate_sample, cache_value=True):
        value_prompt = task.value_prompt_wrap(x, y)
        if cache_value and value_prompt in task.value_cache:
            return task.value_cache[value_prompt]
        value_outputs = self.model(value_prompt, n=n_evaluate_sample, stop=None)
        value = task.value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            task.value_cache[value_prompt] = value
        return value

    def get_values(self, task, x, ys, n_evaluate_sample, cache_value=True):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:    
                value = self.get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                local_value_cache[y] = value
            values.append(value)
        return values

    def get_votes(task, x, ys, n_evaluate_sample):
        vote_prompt = task.vote_prompt_wrap(x, ys)
        vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
        values = task.vote_outputs_unwrap(vote_outputs, len(ys))
        return values

    def get_proposals(task, x, y): 
        propose_prompt = task.propose_prompt_wrap(x, y)
        proposals = model(propose_prompt, n=1, stop=None)[0].split('\n')
        return [y + _ + '\n' for _ in proposals]

    def get_samples(self, task, x, y, n_generate_sample, prompt_sample, stop):
        if prompt_sample == 'standard':
            prompt = task.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = task.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        samples = self.model(prompt, n=n_generate_sample, stop=stop)
        return [y + _ for _ in samples]

    def solve(self, args, task, idx, to_print=True):
        print(model)
        print(f'idx: {idx}')
        print(f'idx: {task}')
        x = task.get_input(args.problem)  # input
        print(f'x: {x}')
        ys = ['']  # current output candidates
        infos = []
        for step in range(task.steps):
            # generation
            if args.method_generate == 'sample':
                new_ys = [self.get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            elif args.method_generate == 'propose':
                new_ys = [self.get_proposals(task, x, y) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            if args.method_evaluate == 'vote':
                values = self.get_votes(task, x, new_ys, args.n_evaluate_sample)
            elif args.method_evaluate == 'value':
                values = self.get_values(task, x, new_ys, args.n_evaluate_sample)

            # selection
            if args.method_select == 'sample':
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
            elif args.method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if to_print: 
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
            
            infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys
        
        if to_print: 
            print(ys)
        return ys, {'steps': infos}

    def naive_solve(self, args, task, idx, to_print=True):
        x = task.get_input(idx)  # input
        ys = self.get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
        return ys, {}

    def run(self, args):
        task = self.get_task(args.task, args.task_file_path)
        logs, cnt_avg, cnt_any = [], 0, 0
        global model
        model = partial(self.model, model=args.backend, temperature=args.temperature)
        if args.naive_run:
            file = f'logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
        else:
            file = f'logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
        os.makedirs(os.path.dirname(file), exist_ok=True)

        for i in range(args.task_start_index, args.task_end_index):
            # solve
            if args.naive_run:
                ys, info = self.naive_solve(args, task, i) 
            else:
                ys, info = self.solve(args, task, i)

            # log
            infos = [task.test_output(i, y) for y in ys]
            info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
            logs.append(info)
            with open(file, 'w') as f:
                json.dump(logs, f, indent=4)
            
            # log main metric
            accs = [info['r'] for info in infos]
            cnt_avg += sum(accs) / len(accs)
            cnt_any += any(accs)
            print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
        
        n = args.task_end_index - args.task_start_index
        print(cnt_avg / n, cnt_any / n)
        print('usage_so_far', gpt_usage(args.backend))




            












#does not output state after each thought --- idk why -- needs work
class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        print(f'Start time {start_time}')
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                print(f'result in optimized tree of thoughts: {result}')
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
    
    #create instance
    parser = argparse.ArgumentParser(description="Tree of Thoughts Solver")
    parser.add_argument("--problem", type=str, required=True, help="Initial problem statement")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1, help="Version of Tree of Thoughts to use (v1 or v2)")


    # input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"

    # parser.add_argument("--problem", type=str, required=True, help="Initial problem statement")
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


    #args for v1 original paper implementation
    parser.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-4')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--task', type=str, required=False, choices=['game24', 'text', 'crosswords'])
    parser.add_argument('--task_file_path', type=str, required=False)
    parser.add_argument('--task_start_index', type=int, default=900)
    parser.add_argument('--task_end_index', type=int, default=1000)
    parser.add_argument('--naive_run', action='store_true')
    parser.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run
    parser.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    parser.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    parser.add_argument('--method_select', type=str, choices=['sample', 'greedy'])
    parser.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    parser.add_argument('--n_evaluate_sample', type=int, default=1)
    parser.add_argument('--n_select_sample', type=int, default=1)




    #args from original implementation

    args = parser.parse_args()
    print(args)
    





    # Initialize the Tree of Thoughts based on the provided version
    if args.version == 1:
        # Initialize the model
        model = partial(gpt, model=args.backend, temperature=args.temperature)

        # Initialize the task with the user's input
        task = TextTask(args.problem)

        # model = chatgpt()
        tree_of_thoughts = TreeofThoughtsv1(model)
        task = None  # Replace with the actual task instance
        idx = 0 # Since we have only one input, set the index to 0
        solution, info = tree_of_thoughts.solve(args, task, idx)
        print(f'Solutions: {solution}')
        # solution = TreeofThoughtsv1.solve(args.problem)
        # print(f'Solutions: {solution}')
    elif args.version == 2:
        model = OptimizedOpenAILanguageModel(os.getenv('OPENAI_API_KEY'), api_model="gpt4")
        #solve the problem using the tree of thoughts class
        optimized_tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm=args.search_algorithm)

        #solve the porblem using tree of thoughts problem helper
        best_state = optimized_tree_of_thoughts.solve(args.problem, k=args.k, T=args.T, b=args.b, vth=args.vth)


        #generate the final silution
        final_solution = optimized_tree_of_thoughts.model.generate_solution(best_state, args.problem)


        #print the final solutions
        print(f"Final solution: {final_solution}")
