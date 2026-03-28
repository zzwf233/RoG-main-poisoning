from typing import List, Tuple

import transformers


def align_tokenizer_vocab_with_model(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    base_special_tokens: List[str],
    pad_token_prefix: str = "<ROG_EXTRA_SPECIAL_>",
) -> Tuple[int, int, int]:
    """
    Keep slow tokenizer + model vocab sizes aligned by:
    1) Adding required RoG special tokens.
    2) If model vocab is still larger than tokenizer length, padding tokenizer with
       deterministic extra special tokens until lengths match.
    3) Resizing model embeddings to tokenizer length.

    Returns:
        (added_base_tokens, added_padding_tokens, final_tokenizer_len)
    """
    added_base_tokens = tokenizer.add_tokens(base_special_tokens)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_len = len(tokenizer)
    added_padding_tokens = 0
    if tokenizer_len < model_vocab_size:
        missing = model_vocab_size - tokenizer_len
        filler_tokens = [f"{pad_token_prefix}{i}>" for i in range(missing)]
        added_padding_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": filler_tokens}
        )

    model.resize_token_embeddings(len(tokenizer))
    return added_base_tokens, added_padding_tokens, len(tokenizer)
