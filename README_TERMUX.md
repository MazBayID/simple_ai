# Test Simple AI on Termux (Android)

Simple AI is an offline AI research prototype built from scratch. It is not
yet a reliable assistant.

## 1. Install tools

```bash
pkg update
pkg upgrade
pkg install python git clang cmake
python -m pip install pytest
```

## 2. Update and test

```bash
cd ~/simple_ai
git pull origin main
PYTHONPATH=python python -m pytest -q
```

## 3. Collect and prepare the Indonesian corpus

Download and prepare the rights-cleared Indonesian Wikipedia subset (CC BY-SA 4.0).
The script checks each record's license and revision provenance, normalizes text,
removes duplicate documents, and creates deterministic document-level splits.
Raw and processed data remain local and are ignored by Git.

    PYTHONPATH=python python -m simple_ai.experiments.prepare_indonesian_corpus

It requires at least 1,000,000 training characters. See
docs/INDONESIAN_CORPUS.md for source, attribution, manifest, and license details.

## 4. First million-character training benchmark

Start with one epoch because the pure-Python trainer is not optimized for a
large corpus. The test split is evaluated after training and is not used for
gradient updates.

    PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --input data/processed/indonesian_wikipedia/train.txt --validation-input data/processed/indonesian_wikipedia/validation.txt --test-input data/processed/indonesian_wikipedia/test.txt --tokenizer bpe --vocab-size 256 --architecture mlp --context-size 32 --embedding-dim 8 --hidden-size 32 --epochs 1 --batch-size 64 --output-dir artifacts/indonesian_wikipedia_baseline

## 5. Run the local chat harness

    PYTHONPATH=python python -m simple_ai.chat --model artifacts/indonesian_wikipedia_baseline/model.json --tokenizer artifacts/indonesian_wikipedia_baseline/tokenizer.json

## 6. Train the small Indonesian seed baseline

Default: BPE tokenizer, MLP architecture, vocabulary target 128, context 32,
20 epochs. The actual BPE vocabulary may be smaller if there are not enough
repeated pairs in the corpus. The MLP uses SGD by default; Transformer training
uses Adam with a smaller default learning rate.

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

The model, tokenizer, and report are saved under
`artifacts/indonesian_model/`.

## 7. Try the causal Transformer

Start small because this pure-Python implementation is slower than the MLP:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --vocab-size 128 \
  --context-size 16 --embedding-dim 8 --epochs 5 --batch-size 8
```

This overwrites the default Indonesian checkpoint. Keep its model and tokenizer
together, then run:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

## 8. Try the multi-head Transformer v2

The original `--architecture transformer` remains available as a single-head
baseline. The new `transformer_v2` adds multiple causal attention heads,
pre-layer normalization, residual connections, and a final normalization.
It uses a separate checkpoint format so the original model remains loadable.

Start with a small, controlled run:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --input artifacts/benchmark_inputs/train.txt \
  --validation-input artifacts/benchmark_inputs/validation.txt \
  --tokenizer bpe --vocab-size 151 --architecture transformer_v2 \
  --context-size 32 --embedding-dim 8 --num-heads 2 \
  --epochs 1 --batch-size 64 \
  --output-dir artifacts/transformer_v2_benchmark
```

Compare `final_validation_loss`, `final_validation_perplexity`, training
time, and generated samples against the MLP and original Transformer runs.
Keep the same tokenizer, data split, seed, and training settings when comparing
architectures. One run is only a smoke benchmark, not proof of better language
quality.

## 9. Compare character tokenization

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

## 10. Train from your own data

Plain UTF-8 text:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --input /path/to/indonesian_corpus.txt --tokenizer bpe --vocab-size 512
```

JSONL dialogue records (one object per line with non-empty `user` and
`assistant` strings):

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --conversations data/conversations.example.jsonl
```

For a separate validation corpus (recommended for comparisons), add
`--validation-input /path/to/validation.txt`. The tokenizer is fitted only on
the training file; unseen validation characters map to the unknown token.

Only use data you have permission to train on. Keep private corpora outside Git.

## 11. Benchmark inference

After training, measure model loading, generation speed, checkpoint size, and
Python-traced peak memory:

```bash
PYTHONPATH=python python -m simple_ai.experiments.benchmark_model \
  --generation-tokens 32 --prompt "Halo, apa kabar?"
```

The memory metric is Python allocations, not total Android process RAM.

## 12. Evaluate generated text

After training, run the same fixed prompts against a checkpoint. The report
records the architecture, active compute backend, generation speed, token
diversity, and generated samples. These are diagnostics, not automatic
semantic-quality scores; read the samples and judge relevance, fluency,
factuality, and repetition yourself.

```bash
PYTHONPATH=python python -m simple_ai.experiments.evaluate_generation \
  --model artifacts/indonesian_wikipedia_baseline/model.json \
  --tokenizer artifacts/indonesian_wikipedia_baseline/tokenizer.json \
  --max-new-tokens 80 \
  --output artifacts/evaluation/indonesian_generation.json
```

You can provide your own prompts by repeating `--prompt`, or pass a UTF-8
file with one prompt per line using `--prompts-file`.

Training reports and inference benchmarks now include `compute_backend`
(`numpy` or `python`) so results can be compared without guessing which
matrix multiplication path was active.

## 13. Build the C++ prototype

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## 14. Audit corpus character composition

Before changing the tokenizer or filtering data, inspect the Unicode
composition of each split. The audit is read-only and does not delete
non-Latin text automatically:

```bash
PYTHONPATH=python python -m simple_ai.experiments.audit_corpus \
  --input data/processed/indonesian_wikipedia/train.txt \
  --output artifacts/evaluation/corpus_unicode_audit.json
```

Repeat for `validation.txt` and `test.txt`. Non-Latin characters are
reported for review, not automatically treated as errors.

## Important limitations

- BPE is an educational Unicode-character merge tokenizer, not a production tokenizer.
- The Transformer is one block and one attention head; it is a learning prototype.
- The chat is a text-generation harness, not proof of comprehension.
- Validation can be a held-out text region, held-out JSONL records, or a separate corpus; none replaces an untouched final test set.
- RAM/CPU limits remain deferred until we benchmark on the target Android device.
