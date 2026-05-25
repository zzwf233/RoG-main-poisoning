import time
import os
import openai
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--api_base', type=str, default=os.getenv("OPENAI_BASE_URL", ""))
        parser.add_argument('--api_timeout', type=float, default=30.0)
        parser.add_argument('--max_new_tokens', type=int, default=128)
        parser.add_argument('--temperature', type=float, default=0.0)
        parser.add_argument('--top_p', type=float, default=1.0)
    
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150 
        self.client = None
    def tokenize(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except KeyError:
            raise KeyError(f"Warning: model {self.model_name} not found.")
        return num_tokens + self.redundant_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        '''
        ChatGPT model does not need to prepare for inference
        '''
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for ChatGPT inference.")

        if hasattr(openai, "OpenAI"):
            client_kwargs = {
                "api_key": api_key,
                "timeout": self.args.api_timeout,
            }
            if self.args.api_base:
                client_kwargs["base_url"] = self.args.api_base
            self.client = openai.OpenAI(**client_kwargs)
        else:
            openai.api_key = api_key
            if self.args.api_base:
                openai.api_base = self.args.api_base
            self.client = openai
    
    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.tokenize(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        last_error = None
        while cur_retry <= num_retry:
            try:
                if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=query,
                        max_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                    )
                    result = response.choices[0].message.content.strip()
                else:
                    response = self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=query,
                        max_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        request_timeout=self.args.api_timeout,
                    )
                    result = response["choices"][0]["message"]["content"].strip() # type: ignore
                return result
            except Exception as e:
                last_error = e
                if cur_retry >= num_retry:
                    break
                print(f"API request failed ({cur_retry + 1}/{num_retry + 1}): {e}")
                time.sleep(min(30, 2 ** cur_retry))
                cur_retry += 1
                continue
        print("Message: ", llm_input)
        print("Number of token: ", self.tokenize(llm_input))
        print(last_error)
        return None
