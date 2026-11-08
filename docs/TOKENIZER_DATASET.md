# Tokenizer and Dataset Pipeline

## Purpose

This milestone introduces the first text-to-token data path for Simple AI.
It is a correctness-first baseline for experiments, not a production language
tokenizer and not a trained language model.

## Character tokenizer

`simple_ai.tokenizer.CharTokenizer` maps each Unicode character to a stable
integer ID. IDs 0–3 are reserved for:

- `<pad>`
- `<unk>`
- `<bos>`
- `<eos>`

The remaining vocabulary is sorted by Unicode code point, so building from
the same set of characters produces deterministic IDs. Encoding adds BOS and
EOS by default. Unknown characters map to UNK. Decoding can omit special
tokens.

This baseline is intentionally character-level. It can represent Indonesian
text and other Unicode characters, but sequences are longer than with
subword tokenization. A later milestone can compare BPE or another subword
method using measured vocabulary size, sequence length, and memory use.

## Next-token dataset

`simple_ai.dataset.NextTokenDataset` creates fixed-length context/target
pairs. Given token IDs and context size N, each example contains N consecutive
input IDs and the immediately following target ID. Incomplete windows are
discarded. The batch iterator can shuffle indices reproducibly with a seed
and keeps the final partial batch.

The dataset currently returns Python integer lists. It does not yet create
Tensor objects or train a language model; embedding and a vocabulary-sized
output layer are future work.

## Prepare a local corpus

From the repository root:

```bash
PYTHONPATH=python python -m simple_ai.pipeline --input sample.txt --context-size 16
```

Outputs are written to `artifacts/text_pipeline/`:

- `tokenizer.json` — vocabulary and special-token definition.
- `token_ids.json` — encoded corpus, including BOS and EOS.
- `dataset.json` — metadata and example count.

The source file is read as UTF-8 and is never modified. The pipeline rejects
empty, whitespace-only, and too-short corpora with a clear error. Use a
smaller context size for a short corpus.

## Data handling and limitations

Local `sample.txt` and generated `artifacts/` are ignored by Git by default.
This prevents accidental commits of a user's private corpus and machine-local
checkpoints. Only add data to the repository deliberately, after checking its
license, privacy, and suitability.

The initial pipeline uses the entire supplied text as one corpus. It does not
yet split training/validation/test sets, deduplicate documents, filter
sensitive data, or normalize text. Those controls must be designed before
training on a larger corpus. Never treat a tiny sample as sufficient training
data for a useful assistant.
