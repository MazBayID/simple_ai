"""Generate text from a trained Simple AI v2 PyTorch checkpoint."""
from __future__ import annotations

import argparse
import torch

from ..tokenizer import load_tokenizer
from ..torch_backend.checkpoint import load_checkpoint
from ..torch_backend.model import GPTConfig, GPTLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Indonesian text with a Simple AI v2 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("Generation length, temperature, and top-k must be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    # Checkpoints are Python serialization files: only load trusted project files.
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if payload.get("format") != "simple-ai-torch-checkpoint":
        raise SystemExit("Unsupported checkpoint format.")
    config_data = payload.get("config", {}).get("model")
    if not isinstance(config_data, dict):
        raise SystemExit("Checkpoint does not contain a model configuration.")

    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != config_data.get("vocab_size"):
        raise SystemExit("Tokenizer vocabulary does not match checkpoint.")

    model = GPTLanguageModel(GPTConfig(**config_data)).to(device)
    load_checkpoint(
        args.checkpoint, model=model, map_location=device, restore_rng=False
    )
    model.eval()

    # A prompt is an unfinished document: BOS yes, EOS no.
    prompt_ids = tokenizer.encode(
        args.prompt, add_bos=True, add_eos=False
    )
    if len(prompt_ids) > model.config.context_size:
        prompt_ids = prompt_ids[-model.config.context_size:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
    )
    new_ids = generated[0, len(prompt_ids):].tolist()
    continuation = tokenizer.decode(new_ids)
    print(continuation)


if __name__ == "__main__":
    main()
