import os
import json
import itertools
import argparse
import numpy as np
from functools import partial
from models import gpt, gpt_usage
from tasks import get_task

class CustomLanguageModel:
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        # Implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        # Implement state evaluation logic using self.model
        pass


class TreeofThoughtsV2:
    def __init__(self, args, model):
        self.args = args
        self.model = CustomLanguageModel(model)

    def get_value(self, task, x, y, n_evaluate_sample, cache_value=True):
        value_prompt = task.value_prompt_wrap(x, y)
        if cache_value and value_prompt in task.value_cache:
            return task.value_cache[value_prompt]
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
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
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
        return values


    def get_votes(self, task, x, ys, n_evaluate_sample):
        # ...

    def get_proposals(self, task, x, y):
        # ...

    def get_samples(self, task, x, y, n_generate_sample, prompt_sample, stop):
        # ...

    def solve(self, task, idx, to_print=True):
        # ...

    def naive_solve(self, task, idx, to_print=True):
        # ...

    def run(self):


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_file_path', type=str, required=True)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'])
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args
