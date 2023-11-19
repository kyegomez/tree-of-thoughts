from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Union

import requests


def _default_extractor(json_response: Dict[str, Any], stop_parameter_name) -> str:
    """
    This function extracts the response from the JSON object using the default parameter name.

    Parameters:
        json_response (dict): The JSON response to be extracted.
        stop_parameter_name (str): The name of the parameter that stops the extraction process.

    Returns:
        str: The extracted response.
    """
    return json_response["response"]


@dataclass
class _HTTPBaseLLM:
    prompt_url: str
    parameters: Dict[str, Union[float, int, str, bool, List[str]]] = None
    response_extractor: Callable[[Dict[str, Any]], str] = _default_extractor
    stop_parameter_name: str = "stop"

    @property
    def _llm_type(self) -> str:
        return "custom"

    def sample_n(self, prompt, stop, n):
        samples = []
        for _ in range(n):
            samples.append(self._call(prompt, stop))
        return samples

    def _call(self, prompt: str, stop: List[str]) -> str:
        # Merge passed stop list with class parameters
        stop_list = list(
            set(stop).union(set(self.parameters[self.stop_parameter_name]))
        )

        params = deepcopy(self.parameters)
        params[self.stop_parameter_name] = stop_list

        response = requests.post(
            self.prompt_url,
            json={
                "prompt": prompt,
                **params,
            },
        )
        response.raise_for_status()
        return self.response_extractor(
            response.json(), params[self.stop_parameter_name]
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def set_parameter(self, parameter_name, parameter_value):
        self.parameters[parameter_name] = parameter_value


def ui_default_parameters():
    return {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.001,
        "top_p": 0.3,
        "typical_p": 1,
        "repetition_penalty": 1.2,
        "top_k": 30,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1.5,
        "early_stopping": False,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }


def _response_extractor(json_response, stopping_strings):
    """Extract relevant information from the given JSON response."""
    result = json_response["results"][0]["text"]
    for stop_string in stopping_strings:
        # The stop strings from text-generation-webui come back without the last char
        ss = stop_string[0:-1]
        if ss in result:
            cut_result = result.split(ss)[0]
            return cut_result
    return result


def build_text_generation_web_ui_client_llm(
    prompt_url="http://0.0.0.0:5000/api/v1/generate", parameters=None
):
    """
    This function builds a text generation web UI client using LLM (Largue Language Machine) API.
    It takes a URL for the LLM server and optional parameters as inputs.
    If parameters are not provided, default ones will be used. The function returns an HTTP client that can generate text based on user input.

    Parameters:
    prompt_url (str): URL of the LLM server.
    parameters (Optional[dict]): Optional parameters to pass to the LLM API.

    Returns:
    An HTTP client object that can generate text based on user input.
    """

    if parameters is None:
        parameters = ui_default_parameters()

    return _HTTPBaseLLM(
        prompt_url=prompt_url,
        parameters=parameters,
        stop_parameter_name="stopping_strings",
        response_extractor=_response_extractor,
    )
