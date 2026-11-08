# Training data

- `indonesian_seed.txt` is a small original starter corpus for smoke tests only.
- `conversations.example.jsonl` demonstrates the supported dialogue format.
- Put larger private corpora in `data/private/` or `data/local/`; those paths are
  ignored by Git. You can also keep corpora anywhere on the device and pass their
  absolute path to `--input` or `--validation-input`.
- Use only material you created, licensed, or have explicit permission to use.
- Keep train, validation, and final test corpora separate. Do not tune repeatedly
  against the final test set.
- Clean duplicates, broken encodings, boilerplate, and low-quality text before
  training. Record the source, license, collection date, and preprocessing steps
  in a local note that is not committed if it contains private details.

The starter corpus is not large enough to teach broad Indonesian language ability.
