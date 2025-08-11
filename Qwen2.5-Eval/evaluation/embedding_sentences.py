# Requires transformers>=4.51.0

import argparse
import json
import os
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def read_code_sentences_from_jsonl(jsonl_path: str) -> List[str]:
    sentences: List[str] = []
    sentences_prompt: List[str] = []
    prompt_id: int = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = obj.get("code")
            if isinstance(code, list):
                # Keep order; filter to strings only
                sentences.extend([s for s in code if isinstance(s, str) and s.strip()])
                sentences_prompt.extend([f"Prompt {prompt_id}: {s}" for s in code if isinstance(s, str) and s.strip()])
                prompt_id += 1

    return sentences, sentences_prompt


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_sentences(
    sentences: List[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 64,
    max_length: int = 2048,
    device: str = "auto",
    dtype: str = "auto",
    normalize: bool = True,
    use_flash_attn: bool = False,
) -> Tuple[Tensor, List[str]]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    model_kwargs = {"torch_dtype": torch_dtype}
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    all_embeds: List[Tensor] = []

    with torch.no_grad():
        for chunk in batch_iter(sentences, batch_size):
            batch_dict = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])  # type: ignore[attr-defined]
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeds.append(embeddings.to("cpu"))

    return torch.cat(all_embeds, dim=0), sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed sentences from JSONL 'code' fields using Qwen3 Embedding")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output",
        default="",
        help="Path to save a .pt file with {'sentences': List[str], 'embeddings': Tensor}. Defaults to <input>.embeddings.pt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype to use",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization of embeddings")
    parser.add_argument("--flash-attn", action="store_true", help="Enable flash_attention_2 if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        output_path = f"{os.path.dirname(input_path)}/embeddings.pt"

    sentences, sentences_prompt = read_code_sentences_from_jsonl(input_path)
    print(f"Found {len(sentences)} sentences")
    print(f"Found {len(sentences_prompt)} sentences_prompt")
    if not sentences:
        print(f"No sentences found in 'code' fields: {input_path}")
        return

    embeddings, sentences_out = embed_sentences(
        sentences,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        normalize=not args.no_normalize,
        use_flash_attn=args.flash_attn,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"sentences": sentences_out, "embeddings": embeddings, "sentences_prompt": sentences_prompt}, output_path)
    print(f"Saved embeddings: {embeddings.shape} -> {output_path}")


if __name__ == "__main__":
    main()


