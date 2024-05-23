import logging
from swarms import Agent

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToTAgent:
    """

    OpenAI Language Model API Wrapper

    Args:
        agent (Agent): Agent class from swarms
        strategy (str): Strategy to use for generating thoughts
        evaluation_strategy (str): Strategy to use for evaluating states
        enable_react (bool): Enable ReAct prompting
        k (int): Number of thoughts to generate

    Methods:
        run(task: str) -> list: Generate text from prompt using OpenAI API
        generate_thoughts(state, k, initial_prompt, rejected_solutions=None) -> list: Generate thoughts from state using OpenAI API
        generate_solution(initial_prompt, state, rejected_solutions=None) -> str: Generate solution from state using OpenAI API
        evaluate_states(states, initial_prompt) -> dict: Evaluate states of reasoning using OpenAI API

    Examples:
        >>> from tree_of_thoughts.tot_agent import ToTAgent
        >>> from swarms import Agent
        >>> agent = Agent()
        >>> model = ToTAgent(agent)
        >>> thoughts = model.run("Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'.")
        >>> print(thoughts)
        ['Observation:xxxx', 'Thoughts:xxxx']

    """

    def __init__(
        self,
        agent: Agent,
        strategy: str = "cot",
        evaluation_strategy: str = "value",
        enable_react: bool = True,
        k: int = 3,
        *args,
        **kwargs,
    ):
        self.agent = agent
        self.use_chat_api = True
        self.enable_react = enable_react
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.k = k

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ""
        if enable_react:
            self.react_prompt = (
                "Write down your observations in format 'Observation:xxxx',"
                " then write down your thoughts in format 'Thoughts:xxxx'."
            )

    def run(self, task: str):
        """Generate text from prompt using"""
        if self.use_chat_api:
            thoughts = []
            for _ in range(self.k):
                response = self.agent(task)
                thoughts += [response]
            return thoughts

    def generate_thoughts(
        self, state, k: int = None, initial_prompt: str = None, rejected_solutions: list = None
    ):
        """
        Generate thoughts from state using OpenAI API

        Args:
            state (str or list): State of reasoning
            k (int): Number of thoughts to generate
            initial_prompt (str): Initial prompt
            rejected_solutions (list): List of rejected solutions

        Returns:
            list: List of thoughts



        """
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        print("New state generating thought:", state, "\n\n")
        prompt = f"""Y
        ou're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
        Considering the reasoning provided:\n\n
        ###'{state_text}'\n\n###
        Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
        ###{rejected_solutions}###, 
        complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""

        prompt += self.react_prompt
        thoughts = self.run(prompt)
        return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            answer = self.run(prompt)
            print(f"Answerrrrrr {answer}")
            # print(thoughts)
            # print(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                print(
                    "We receive a state of type",
                    type(state),
                    "For state: ",
                    state,
                    "\n\n",
                )
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """

                response = self.agent(prompt)
                try:
                    value_text = self.openai_choice2text_handler(
                        response.choices[0]
                    )
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == "vote":
            states_text = "\n".join([" ".join(state) for state in states])
            prompt = (
                "Given the following states of reasoning, vote for the best"
                " state utilizing an scalar value"
                f" 1-10:\n{states_text}\n\nVote, on the probability of this"
                f" state of reasoning achieveing {initial_prompt} and become"
                " very pessimistic very NOTHING ELSE"
            )
            response = self.agent(prompt)
            print(f"state response: {response}")
            best_state_text = self.agent(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            print(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError(
                "Invalid evaluation strategy. Choose 'value' or 'vote'."
            )
