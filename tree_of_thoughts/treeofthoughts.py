

import os
import time
import json
from tree_of_thoughts.openaiModels import OptimizedOpenAILanguageModel
DATA_PATH = './data'
import logging 
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = './data'


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
            "nodes": {}
        }

    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        self.file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"
        try:
            best_thoughts = ""
            if self.search_algorithm == 'BFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_bfs(x, k, T, b, vth)
                    if result:
                        self.save_tree_to_json(self.file_name )
                        best_thoughts = result
            elif self.search_algorithm == 'DFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_dfs(x, k, T, vth)
                    if result:
                        self.save_tree_to_json(self.file_name)
                        best_thoughts = result
            if(best_thoughts):
                solution = self.model.generate_solution(x, best_thoughts)
                if solution:
                    return solution
            else:
                raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
        except KeyboardInterrupt:
            logger.error("Keyboard interrupt detected.")
        except ValueError as e:
            logger.error(f"Error: {e}")
        finally:
            logger.info("Saving the current tree and metrics.")
            self.save_tree_to_json(self.file_name)

    def tot_bfs(self, x, k, T, b, pruning_threshold):
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

            for s, v in Vt.items():
                if not (type(s) == str):
                    s = " | ".join(s)
                self.tree["nodes"][s] = v
            print("Saving tree")
            self.save_tree_to_json(self.file_name)
            pruned_S0_t = {s: v for s, v in Vt.items() if v >= pruning_threshold}

            St = sorted(pruned_S0_t.keys(), key=lambda s: pruned_S0_t[s], reverse=True)[:b]
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

                    if dfs(child, t + 1):
                        return True

            self.save_tree_to_json(file_name)
            return False
        
            
        dfs(x, 1)
        print(f'output  {output}')
        best_state = max(output, key=lambda x: x[1])
        return best_state[0]


    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def print_tree(self, node, depth=0):
        thought = self.tree["metrics"]["thoughts"].get(node, "")
        evaluation = self.tree["metrics"]["evaluations"].get(node, "")

        tree_info = f"{'  ' * depth}Node: {node}, Thought: {thought}, Evaluation: {evaluation}\n"

        for child, parent in self.tree["nodes"].items():
            if parent == node:
                tree_info += self.print_tree(child, depth + 1)
                print(f'tree info: {tree_info}')

        return tree_info



#does not output state after each thought --- idk why -- needs work
class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        print(f'Start time {start_time}')
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b, pruning_threshold=0.5)
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


    #args from original implementation

    args = parser.parse_args()
    print(args)
    
    model = OptimizedOpenAILanguageModel()
    #solve the problem using the tree of thoughts class
    optimized_tree_of_thoughts = TreeofThoughts(model, search_algorithm=args.search_algorithm)

    #solve the porblem using tree of thoughts problem helper
    best_state = optimized_tree_of_thoughts.solve(args.problem, k=args.k, T=args.T, b=args.b, vth=args.vth)


    #generate the final silution
    final_solution = optimized_tree_of_thoughts.model.generate_solution(best_state, args.problem)


    #print the final solutions
    print(f"Final solution: {final_solution}")
