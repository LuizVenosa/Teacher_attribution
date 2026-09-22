"""Read declared context limits without confusing long contexts with HF sentinels."""


def context_limit(config, tokenizer) -> int:
    text_config = getattr(config, "text_config", config)
    limits = []
    for key in ("max_position_embeddings", "n_positions"):
        value = getattr(text_config, key, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            limits.append(value)
    # Tokenizers without a bound use an approximately 1e30 sentinel.
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**20:
        limits.append(value)
    if not limits:
        raise ValueError("No explicit model/tokenizer context limit found; inspect model config")
    return min(limits)
