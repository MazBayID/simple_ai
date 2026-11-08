# Test Simple AI on Termux (Android)

Simple AI is an offline AI research project built from scratch. The current language model is a small character-level MLP experiment, not a capable assistant.

## 1. Install tools

```bash
pkg update
pkg upgrade
pkg install python git clang cmake
python -m pip install pytest
```

## 2. Clone or update

```bash
git clone https://github.com/MazBayID/simple_ai.git
cd simple_ai
```

For an existing clone:

```bash
cd ~/simple_ai
git pull origin main
```

## 3. Run all tests

```bash
PYTHONPATH=python python -m pytest -q
```

## 4. Run the original toy demo and regression smoke test

```bash
PYTHONPATH=python python -m simple_ai.cli demo
PYTHONPATH=python python -m simple_ai.experiments.train_regression
```

These validate the early model and neural training engine, not Indonesian language ability.

## 5. Train the Indonesian baseline

The repository includes a small, original Indonesian seed corpus so the entire pipeline can run immediately:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Default settings use context size 32 and 20 epochs. The run prints training and validation loss and writes:

- `artifacts/indonesian_model/model.json`
- `artifacts/indonesian_model/tokenizer.json`
- `artifacts/indonesian_model/training_report.json`

You can train directly from your own UTF-8 text file:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --input sample.txt --context-size 32 --epochs 20
```

Or use JSONL conversation pairs. Each line must be a JSON object with non-empty `user` and `assistant` strings:

```json
{"user":"Apa itu Python?","assistant":"Python adalah bahasa pemrograman."}
{"user":"Apa fungsi variabel?","assistant":"Variabel menyimpan nilai."}
```

Example command:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --conversations data/conversations.example.jsonl --context-size 48 --epochs 80
```

## 6. Try the experimental terminal chat

After training successfully:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

Type `/exit` to quit. Optional settings:

```bash
PYTHONPATH=python python -m simple_ai.chat --max-new-tokens 120 --temperature 0.7 --top-k 8
```

The chat interface loads the local checkpoint; it does not train during chatting. The model may produce nonsense, repeat phrases, or fail to answer. A small seed corpus is only an end-to-end smoke test. Use a much larger, legally obtained Indonesian corpus and diverse conversation examples before judging language ability.

## 7. Prepare a separate text pipeline

```bash
PYTHONPATH=python python -m simple_ai.pipeline --input sample.txt --context-size 16
```

This creates files in `artifacts/text_pipeline/` without modifying the source file.

## 8. Build the C++ prototype

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## Important limitations

- The tokenizer works per Unicode character, not word or subword.
- The model is a fixed-context MLP, not a Transformer.
- The chat loop is an experimental generation interface, not proof of comprehension.
- Characters absent from training vocabulary map to `<unk>`.
- Validation loss is a rough diagnostic, not a measure of factual accuracy.
- Do not train on private or unlicensed text. Local `artifacts/` and `sample.txt` are ignored by Git.
- RAM/CPU limits remain deferred until we benchmark on the target Android device.
