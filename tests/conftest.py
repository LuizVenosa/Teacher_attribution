import copy
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config(tmp_path):
    path = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    cfg = copy.deepcopy(yaml.safe_load(path.read_text()))
    cfg["run_dir"] = str(tmp_path / "run")
    cfg["data"].update(
        local_dir=str(tmp_path / "sources"),
        datasets=["alpaca"],
        prompts_per_dataset={"train": 6, "val": 4, "test": 4},
    )
    cfg["encoder"].update(
        epochs=2, batch_size=4, patience=1, max_length=32, max_prompt_tokens=8, projection_dim=8
    )
    cfg["evaluation"].update(
        set_repeats=3, bootstrap_repeats=20, set_sizes=[1, 2, 4], logreg_c=[1.0]
    )
    return cfg


@pytest.fixture
def sources(config):
    from teacher_attr.io import write_jsonl

    path = Path(config["data"]["local_dir"]) / "alpaca" / "train.jsonl"
    write_jsonl(
        path,
        [{"id": str(i), "instruction": f"explain item {i} .", "input": ""} for i in range(300)],
    )
    return path


@pytest.fixture
def tiny_models(tmp_path, config):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig,
        BertModel,
        BertTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    torch.set_num_threads(1)
    torch.manual_seed(1)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "explain",
        "item",
        ".",
        "alpha",
        "beta",
        "response",
        "sentinel",
    ] + [str(i) for i in range(300)]
    encoder_dir = tmp_path / "tiny_encoder"
    encoder_dir.mkdir()
    vocab_path = encoder_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(encoder_dir)
    BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    ).save_pretrained(encoder_dir)
    generator_dir = tmp_path / "tiny_generator"
    backend = Tokenizer(WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    generator_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=64
    )
    generator_tokenizer.save_pretrained(generator_dir)
    generator = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=64,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    generator.generation_config.suppress_tokens = [0, 1, 2, 3, 4]
    generator.save_pretrained(generator_dir)
    config["encoder"]["hf_name"] = str(encoder_dir)
    for role in ("teachers", "students"):
        for model in config[role].values():
            model["hf_name"] = str(generator_dir)
    config["generation"].update(max_input_tokens=32, max_new_tokens=3, temperature=0)
    return tokenizer
