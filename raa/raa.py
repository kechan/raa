from typing import List, Dict
import os, logging, openai, tiktoken
import pkg_resources
import openai
try:
  from openai.error import OpenAIError
except ImportError:
  print("Not importing OpenAIError from openai.error, newer version has changed")

from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from dotenv import load_dotenv, find_dotenv


AVAILABLE_GPT_MODELS = ['gpt-3.5-turbo-0613',
                        'gpt-3.5-turbo',
                        'gpt-4',
                        'gpt-4-1106-preview',
                        'gpt-4-0125-preview',
                        'gpt-4o',
                        'gpt-5.6-luna',
                        'gpt-6-luna'
                        ]

logger = logging.getLogger(__name__)

def _is_legacy_model(model) -> bool:
  '''
  gpt-4o and older (gpt-3.5*, gpt-4*) take max_tokens and reject reasoning_effort.
  gpt-5.6-luna and newer are reasoning models: they reject max_tokens and take max_completion_tokens
  (which also counts hidden reasoning tokens) plus reasoning_effort.
  '''
  return model.startswith(('gpt-3', 'gpt-4'))

def _completion_params(model, max_tokens=None) -> dict:
  '''Model-specific kwargs for chat completion calls (everything except model, messages, temperature).'''
  if _is_legacy_model(model):
    return {} if max_tokens is None else {'max_tokens': max_tokens}

  params = {'reasoning_effort': 'low'}
  if max_tokens is not None:
    params['max_completion_tokens'] = max_tokens
  return params

class RAA:
  def __init__(self, llm_model=None, sys_prompt=None, api_key_env_var: str = 'OPENAI_API_KEY'):
    self.llm_model = llm_model
    self.sys_prompt = sys_prompt   # system prompt to be used for all completions, this is the character of the agent.

    # for any openai models
    if not self.llm_model in AVAILABLE_GPT_MODELS:
      raise ValueError(f'llm_model must be one of {AVAILABLE_GPT_MODELS}')

    # Load environment variables
    _ = load_dotenv(find_dotenv()) # read local .env file

    # Set OpenAI API key
    if api_key_env_var in os.environ:
      openai.api_key = os.environ[api_key_env_var]
    else:
      raise ValueError(f'Environment variable {api_key_env_var} not found.')

    # get the installed openai version 
    openai_version = pkg_resources.get_distribution("openai").version
    if pkg_resources.parse_version(openai_version) >= pkg_resources.parse_version("1.3.0"):
      self.openai_version = ">=1.3.0"
      # for newer versions of openai, use the OpenAI class to create a client with the api_key
      from openai import OpenAI
      self.client = OpenAI(api_key = os.environ[api_key_env_var])
    else:
      self.openai_version = "<1.3.0"

    # print(f'openai version: {self.openai_version}')

    self.histories = []

  def test_health(self) -> bool:
    '''
    A quick simple test to see if the openai api endpoint for chat completion is working.
    Use the configured model (or gpt-4o as default), short prompt, and a tiny output cap.
    '''
    try:
      model_name = self.llm_model if self.llm_model is not None else 'gpt-4o'

      messages = [{"role": "system", "content": "Hello"}, {"role": "user", "content": "Hi"}]
      if _is_legacy_model(model_name):
        params = {'temperature': 0.0, 'max_tokens': 2}   # gpt-4o returns truncated text (e.g. "Hello!") at the cap
      else:
        # reasoning_effort='none' is what allows temperature=0.0 (otherwise only 1 is accepted).
        # The cap must fit the whole reply (~10 tokens, "Hi! How can I help?"): unlike gpt-4o, these models
        # raise an error instead of returning truncated text when the cap is hit.
        params = {'temperature': 0.0, 'reasoning_effort': 'none', 'max_completion_tokens': 20}

      if self.openai_version == ">=1.3.0":
        response = self.client.chat.completions.create(model=model_name, messages=messages, **params)
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
      else:
        response = openai.ChatCompletion.create(model=model_name, messages=messages, **params)
        content = response.choices[0].message["content"]
        finish_reason = response.choices[0]["finish_reason"]

      print(content)
      if content is not None and len(content) > 0:   # this could be "Hello!"
        logger.info(f"OpenAI health check ok ({model_name}): {content!r}")
        return True
      else:
        logger.error(f"OpenAI health check ({model_name}) returned empty content, finish_reason={finish_reason}")
        return False
    except Exception as e:
      print(f"Exception: {e}")
      logger.error(f"OpenAI health check ({model_name}) failed: {e}")
      return False

  def get_openai_api_key(self):
    return openai.api_key

  def get_completion(self, prompt, temperature=0):
    """
    Simple one prompt, one completion
    """
    if self.sys_prompt is not None:
      messages = [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": prompt}]
    else:
      messages = [{"role": "user", "content": prompt}]

    if self.llm_model == GPT_MODEL:
      if self.openai_version == ">=1.3.0":
        response = _openai_ChatCompletion_newer_than_1d3d0(client=self.client, messages=messages, model=self.llm_model, temperature=temperature)
      else:
        response = _openai_ChatCompletion_older_than_1d3d0(messages, model=self.llm_model, temperature=temperature)
      return response
    else:
      raise NotImplementedError
    
  def get_completion_from_messages(self, messages, temperature=0, max_tokens=500, debug=False):
    """
    messages is a list of dicts with keys "role" and "content"
    roles can be 'system', 'user', 'assistant'
    E.g. 
    messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Who won the world series in 2020?"},
          {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
          {"role": "user", "content": "Where was it played?"}
      ]
    """
    if messages[0]["role"] != "system":   # system prompt not supplied, try use self.sys_prompt if available
      if self.sys_prompt is not None:
        messages.insert(0, {"role": "system", "content": self.sys_prompt})
   
    if not debug:
      if self.openai_version == '>=1.3.0':
        response = _openai_ChatCompletion_newer_than_1d3d0(client=self.client, messages=messages, model=self.llm_model, temperature=temperature, max_tokens=max_tokens)
      else:
        response = _openai_ChatCompletion_older_than_1d3d0(messages, model=self.llm_model, temperature=temperature, max_tokens=max_tokens)
    else: 
      print(f'llm_model: {self.llm_model}')
      print(f'messages destined for chatgpt:\n{messages}')
      response = "<paste result from chatGPT>"

    return response

  def vectorize_prompt(self, prompt, input_var_name):
    vectorized_prompt = f"""This is a prompt I have designed which takes 1 string as input and it is delimited by <prompt> and </prompt>:
<prompt>
{prompt}
</prompt>

There will be placeholders such as {{blah blah}} which will be provided later. The problem is the prompt is designed to take in one input {{input_var_name}}, 
can you reword the prompt such that I can multiple instances and you output a list of whatever the output format this prompt asked for?
Please provide your answer in the format of {'prompt': "the revised prompt", "revised_input_name": "the revised input name"} 
You may exclude the delimiters <prompt> and </prompt> in the json.
"""
    return vectorized_prompt
  

@retry(wait=wait_random_exponential(min=1, max=40), 
       stop=stop_after_attempt(3), 
       retry_error_cls=RetryError
)
def _openai_ChatCompletion_newer_than_1d3d0(client, messages: List[Dict], model, temperature=0, max_tokens=None) -> str:
  response = None

  try:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **_completion_params(model, max_tokens),
    )
    return response.choices[0].message.content

    # TODO: error handling generalized to particular error messages
  except RetryError as e:
    print("Unable to generate ChatCompletion response after 3 attempts")
    print(f"Exception: {e}")
    raise Exception("Unable to generate ChatCompletion response after 3 attempts")

  except openai.APIError as e:
    # Catching OpenAI specific errors and re-raising them.
    # print(f"OpenAI Error: {e.error['message']}")
    e.chat_messages = messages
    raise e

  except openai.APIConnectionError as e:
    e.chat_messages = messages
    raise e

  except openai.APIStatusError as e:
    e.chat_messages = messages
    raise e
  
  except Exception as e:
    print("An unexpected error occurred while generating ChatCompletion response")
    print(f"Exception: {e}")
    # raise Exception("An unexpected error occurred while generating ChatCompletion response")
    raise e

  
@retry(wait=wait_random_exponential(min=1, max=40), 
       stop=stop_after_attempt(3), 
       retry_error_cls=RetryError
)
def _openai_ChatCompletion_older_than_1d3d0(messages: List[Dict], model, temperature=0, max_tokens=None) -> str:
  response = None

  # use older API
  try:
    # max_tokens None -> no cap passed, assume openai default
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **_completion_params(model, max_tokens),
    )
    return response.choices[0].message["content"]

  # TODO: error handling generalized to particular error messages 
  except RetryError as e:
    print("Unable to generate ChatCompletion response after 3 attempts")
    print(f"Exception: {e}")
    raise Exception("Unable to generate ChatCompletion response after 3 attempts")

  except OpenAIError as e:
    # Catching OpenAI specific errors and re-raising them.
    # print(f"OpenAI Error: {e.error['message']}")
    e.chat_messages = messages
    raise e
  
  except Exception as e:
    print("An unexpected error occurred while generating ChatCompletion response")
    print(f"Exception: {e}")
    # raise Exception("An unexpected error occurred while generating ChatCompletion response")
    raise e
