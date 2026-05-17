import os
import time

from .base_language_model import BaseLanguageModel


class OpenAICompatibleChat(BaseLanguageModel):
    """Chat-completions backend for OpenAI-compatible API providers."""

    @staticmethod
    def add_args(parser):
        parser.add_argument("--api_key", type=str, default="")
        parser.add_argument(
            "--api_base",
            type=str,
            default=os.getenv("OPENAI_BASE_URL", os.getenv("API_BASE", "https://api.siliconflow.cn/v1")),
        )
        parser.add_argument("--api_timeout", type=float, default=60.0)
        parser.add_argument("--retry", type=int, default=5)
        parser.add_argument("--max_new_tokens", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top_p", type=float, default=1.0)
        parser.add_argument("--max_input_chars", type=int, default=60000)

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name
        self.api_key = (
            args.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        self.api_base = args.api_base
        self.retry = args.retry
        self.maximun_token = args.max_input_chars
        self.client = None
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY, SILICONFLOW_API_KEY, "
                "DEEPSEEK_API_KEY, or pass --api_key."
            )

    def load_model(self, **kwargs):
        return None

    def prepare_for_inference(self, **model_kwargs):
        import openai

        # New SDK (openai>=1.x) exposes OpenAI; old SDK uses module globals.
        if hasattr(openai, "OpenAI"):
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.args.api_timeout,
            )
        else:
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            self.client = openai

    def tokenize(self, text):
        # Good enough for prompt truncation in PromptBuilder.
        return max(1, len(str(text)) // 4)

    def generate_sentence(self, llm_input):
        prompt = str(llm_input)
        if len(prompt) > self.args.max_input_chars:
            prompt = prompt[: self.args.max_input_chars]

        messages = [{"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                    )
                    return response.choices[0].message.content.strip()

                response = self.client.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    request_timeout=self.args.api_timeout,
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry:
                    break
                print(f"API request failed ({attempt + 1}/{self.retry + 1}): {exc}")
                time.sleep(min(30, 2 ** attempt))

        print(f"API request failed permanently: {last_error}")
        return None
