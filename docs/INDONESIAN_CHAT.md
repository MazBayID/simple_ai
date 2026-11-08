# Indonesian language and conversation experiment

## Purpose

This milestone connects the existing tokenizer, fixed-context dataset, neural language model, and a small terminal chat harness. It establishes an Indonesian-language training path without importing a pretrained model or a third-party ML framework.

## Data formats

Plain text is accepted as UTF-8. JSONL conversation data contains one object per line:

```json
{"user":"Apa itu Python?","assistant":"Python adalah bahasa pemrograman."}
```

Each record is rendered with the literal role labels `Pengguna:` and `Asisten:`. The labels are ordinary characters, not special tokens and not a true role-aware architecture.

A small Indonesian seed text and six example exchanges are provided only to test the pipeline. They are not a useful-scale training corpus.

## Training and validation

The training command reserves a contiguous tail of the token sequence for validation. A context-sized overlap supplies the first validation example with preceding characters. Training examples come only from the earlier portion. The report records source, corpus size, vocabulary, context size, example counts, parameter count, and initial/final training and validation losses.

This split is a basic diagnostic. It does not replace evaluation on a separate, independently collected test corpus. A falling training loss can indicate memorization. A falling validation loss is not proof of understanding.

## Run

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --conversations data/conversations.example.jsonl
PYTHONPATH=python python -m simple_ai.chat
```

The default output directory is `artifacts/indonesian_model/`.

## Chat loop

The chat loop loads the model and tokenizer, formats a user message with a short history, and autoregressively samples characters. It does not update weights during chat. Unknown input characters become `<unk>`; the model can only generate characters in its trained vocabulary.

## Architecture limitations

The current model is an embedding lookup followed by flattening, one hidden Linear/ReLU layer, and a vocabulary projection. It has a fixed context window and no attention, recurrence, positional sequence processing beyond flattened embedding slots, retrieval, or semantic grounding. The model is a learning baseline—not a dependable conversational assistant.

## Next research milestones

1. Collect a substantially larger Indonesian corpus with clear provenance and redistribution permissions.
2. Build a robust dataset audit: duplicates, encoding, language distribution, unsafe/private content, and train/validation/test separation.
3. Add a subword tokenizer and a more suitable sequence architecture.
4. Evaluate on held-out Indonesian prompts for fluency, relevance, factuality, and consistency.
5. Profile training and inference memory/time on Termux before choosing resource targets.
