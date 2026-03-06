from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # 导入 AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer, LlamaForCausalLM  # 确保导入 LlamaForCausalLM 以备用


class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    ROG_NEW_TOKENS = ["<SEP>", "<PATH>", "</PATH>"]

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path",
                            default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100
        # 🚨 修正 1：在初始化时加载模型，而不是在 prepare_for_inference 中
        # 这确保了模型在 pipeline 之前被加载到 self.model
        self.model = self._load_model_weights()  # 调用内部加载函数

    def _load_model_weights(self):
        # 使用 AutoModelForCausalLM 配合 local_files_only 来加载模型权重
        # 强制使用 LlamaForCausalLM (因为 AutoModelForCausalLM 在推理时可能导致权限问题)
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.args.model_path,
                device_map="auto",
                torch_dtype=self.DTYPE.get(self.args.dtype, None),
                local_files_only=True  # 强制本地加载
            )
            return model
        except Exception as e:
            print(f"Warning: Failed to load LlamaForCausalLM locally. Falling back to AutoModel: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                device_map="auto",
                torch_dtype=self.DTYPE.get(self.args.dtype, None),
                local_files_only=True  # 强制本地加载
            )
            return model

    def load_model(self, **kwargs):
        # 🚨 修正 2：确保分词器加载也使用 local_files_only
        kwargs.update({'local_files_only': True, 'use_auth_token': False})
        kwargs.update({'use_fast': False})

        # 🚨 最终修正：使用 self.args.model_path 作为第一个位置参数，并传入 **kwargs
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, **kwargs)
        return tokenizer

    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        # 🚨 修正 3：修正分词器加载，使用 local_files_only，并传入加载好的模型

        # 1. 加载分词器：直接调用 self.load_model，让它自行处理 self.args.model_path
        self.tokenizer = self.load_model()  # <--- 移除参数，让 load_model 使用 self.args.model_path
        # 与规则生成端保持一致：注入 RoG 特殊 token，并对齐模型 embedding 大小。
        # 否则可能在 decode 阶段出现 piece id is out of range。
        num_added = self.tokenizer.add_tokens(self.ROG_NEW_TOKENS)
        if num_added > 0 or len(self.tokenizer) != self.model.get_input_embeddings().weight.size(0):
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 初始化 pipeline
        # pipeline 不再需要从 args.model_path 下载模型，因为它将接收 self.model
        self.generator = pipeline(
            "text-generation",
            model=self.model,  # 传入预先加载的本地模型对象
            tokenizer=self.tokenizer,
            device_map="auto",
            model_kwargs=model_kwargs,
            torch_dtype=self.DTYPE.get(self.args.dtype, None)
        )

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
        return outputs[0]['generated_text']  # type: ignore
