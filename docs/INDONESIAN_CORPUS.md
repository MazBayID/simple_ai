# Indonesian corpus preparation

Simple AI prepares a rights-cleared subset of Indonesian Wikipedia introductory extracts.

- Dataset: https://huggingface.co/datasets/apricitea/bahasa-indonesia-corpus
- License: CC BY-SA 4.0
- Each record must have rights_reviewed=true, the declared license, an id.wikipedia.org URL, and a revision ID.
- The script normalizes Unicode and whitespace, removes exact duplicate texts, prepends article titles, and splits whole documents deterministically.
- Split: 90% training, 5% validation, 5% test using SHA-256 of source URL + revision ID.
- Raw and processed data remain local and are ignored by Git. manifest.json stores provenance, counts, and hashes.

## Prepare on Termux

    cd ~/simple_ai
    git pull origin main
    PYTHONPATH=python python -m simple_ai.experiments.prepare_indonesian_corpus

The script downloads the dataset if absent and refuses to write outputs unless
the training split contains at least 1,000,000 characters.

Outputs:
- data/processed/indonesian_wikipedia/train.txt
- data/processed/indonesian_wikipedia/validation.txt
- data/processed/indonesian_wikipedia/test.txt
- data/processed/indonesian_wikipedia/manifest.json
- data/processed/indonesian_wikipedia/ATTRIBUTION.md

## First training run

Start with one epoch as a runtime benchmark. The pure-Python trainer is not
optimized for million-character corpora.

    PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --input data/processed/indonesian_wikipedia/train.txt --validation-input data/processed/indonesian_wikipedia/validation.txt --tokenizer bpe --vocab-size 256 --architecture mlp --context-size 32 --embedding-dim 8 --hidden-size 32 --epochs 1 --batch-size 64 --output-dir artifacts/indonesian_wikipedia_baseline

The untouched test split will be evaluated after training when --test-input is
supported by the trainer. Preserve the test split for final evaluation.

## Licensing

Preserve attribution and review share-alike requirements before redistributing
the corpus or trained artifacts, especially for commercial use. This is not
legal advice.
