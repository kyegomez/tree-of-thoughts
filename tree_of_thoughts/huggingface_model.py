from swarms.models import HuggingfaceLLM


class HuggingLanguageModel:
    """
    Initializes a HuggingLanguageModel object.

    Args:
        model_name (str): The name of the Huggingface language model.
        model_tokenizer (object, optional): The tokenizer object for the language model.
        verbose (bool, optional): Flag indicating whether to print verbose output.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self, model_name, model_tokenizer=None, verbose=False, *args, **kwargs
    ):
        self.model = HuggingfaceLLM(model_name, *args, **kwargs)
        self.verbose = verbose

    def generate_thoughts(self, state, k, max_length=100):
        """
        Generates coherent thoughts based on a given state.

        Args:
            state (list): The current state of reasoning.
            k (int): The number of coherent solutions to generate.
            max_length (int, optional): The maximum length of the generated thoughts.

        Returns:
            list: A list of generated thoughts.

        Raises:
            Exception: If there is an error generating thoughts.
        """
        state_text = " ".join(state)
        prompt = (
            "Write down your observations in format 'Observation:xxxx', then"
            " write down your thoughts in format 'Thoughts:xxxx Given the"
            f" current state of reasoning: '{state_text}', generate"
            f" {k} coherent solutions to achieve {state_text}"
        )

        if self.verbose:
            print(f"Generating thoughts for state: {state_text}")

        try:
            self.model.run(prompt)
        except Exception as e:
            if self.verbose:
                print(f"Error generating thoughts for state: {state_text}")
                print(f"Error: {e}")
            thoughts = []

        return thoughts

    def evaluate_states(self, states, initial_prompt, max_length=10):
        """
        Evaluates the value of multiple states based on an initial prompt.

        Args:
            states (list): A list of states to evaluate.
            initial_prompt (str): The initial prompt for evaluation.
            max_length (int, optional): The maximum length of the evaluation.

        Returns:
            dict: A dictionary mapping each state to its evaluated value.

        Raises:
            Exception: If there is an error evaluating states.
        """
        state_values = {}
        for state in states:
            state_text = " ".join(state)
            prompt = (
                f"Given the current state of reasoning: '{state_text}',"
                " pessimitically evaluate its value as a float between 0 and 1"
                f" based on it's potential to achieve {initial_prompt}"
            )

            if self.verbose:
                print(f"Evaluating state: {state_text}")

            try:
                value_text = self.model(prompt)
                value = float(value_text)
            except ValueError:
                if self.verbose:
                    print(
                        "Error converting value to float for state:"
                        f" {state_text}"
                    )
                value = 0  # Assign a default value if the conversion fails
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating state: {state_text}")
                    print(f"Error: {e}")
                value = 0

            state_values[state] = value

        return state_values
