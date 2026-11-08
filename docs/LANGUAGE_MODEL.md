# Tiny Language Model

This milestone connects the text pipeline to the neural engine and trains a
small next-character predictor.

## Architecture

```text
token IDs (batch, context)
        |
trainable embedding lookup
        |
flattened position-aware vectors
        |
Linear -> ReLU -> Linear
        |
vocabulary logits (batch, vocab_size)
        |
cross-entropy against next-token IDs
```

This is a small fixed-context multilayer perceptron (MLP), not a Transformer.
It has no attention, recurrent state, or long-range context mechanism.

## Train on a local corpus

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_language_model \
  --input sample.txt --context-size 8 --epochs 30
```

The command saves `model.json` and `tokenizer.json` under
`artifacts/language_model/`. It prints training loss and a generated sample.

## What generation does

At each step, the model reads the latest fixed-size context, scores every
vocabulary token, samples one token, and appends it to the context. This
autoregressive loop is the first end-to-end language-model inference path.

## Important limitations

- The default `sample.txt` is only a tiny smoke-test corpus. It cannot teach
  broad language understanding or useful conversation.
- Character tokenization makes sequences long and spelling/word learning hard.
- The model sees only a short fixed context and has no attention mechanism.
- The first implementation trains on the supplied corpus without a validation
  split; its loss measures fit to that corpus, not generalization.
- Generated text may be repetitive or incoherent. A falling loss is not proof
  of conversational ability.
- Checkpoints are experimental and are not release-ready model files.
