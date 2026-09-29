"""Explicit response-only likelihood tokenization, including one-token answers."""


def response_tokens(tokenizer, text: str, limit: int, identity: str) -> list[int]:
    start = tokenizer.bos_token_id
    if start is None:
        start = tokenizer.eos_token_id
    if start is None:
        raise ValueError("Perplexity control requires a BOS or EOS boundary token")
    response = tokenizer.encode(text, add_special_tokens=False)
    if not text.strip() or not response:
        raise ValueError(f"{identity}: empty perplexity response")
    tokens = [start, *response]
    if len(tokens) > limit:
        raise ValueError(
            f"{identity}: perplexity input needs {len(tokens)} tokens including the "
            f"boundary token; context budget is {limit}. No truncation applied."
        )
    return tokens
