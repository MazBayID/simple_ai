from simple_ai.dataset import NextTokenDataset
from simple_ai.experiments.train_conversation_model import evaluate
from simple_ai.language_model import TinyLanguageModel
from simple_ai.optim import SGD
from simple_ai.conversation import render_conversations
from simple_ai.tokenizer import CharTokenizer


def test_conversation_text_can_train_model():
    text = render_conversations([
        {"user": "Halo", "assistant": "Hai"},
        {"user": "Apa itu Python?", "assistant": "Bahasa pemrograman."},
    ])
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)
    dataset = NextTokenDataset(ids, context_size=8)
    model = TinyLanguageModel(tokenizer.vocab_size, 8, seed=3)
    optimizer = SGD(model.parameters(), learning_rate=0.02)
    assert len(dataset) > 0
    assert evaluate(model, dataset, 8) > 0


def test_evaluate_empty_dataset_returns_none():
    tokenizer = CharTokenizer.from_text("abc")
    model = TinyLanguageModel(tokenizer.vocab_size, 8)
    assert evaluate(model, NextTokenDataset([], context_size=8), 4) is None


def test_jsonl_validation_splits_whole_records_and_keeps_vocab_train_only(tmp_path, monkeypatch):
    import json
    from simple_ai.experiments.train_conversation_model import main
    from simple_ai.tokenizer import CharTokenizer

    conversations = tmp_path / "pairs.jsonl"
    conversations.write_text(
        '{"user":"Halo","assistant":"Hai"}\\n'
        '{"user":"zebra","assistant":"zebra"}\\n',
        encoding="utf-8",
    )
    output = tmp_path / "record-split"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_conversation_model", "--conversations", str(conversations),
            "--tokenizer", "char", "--context-size", "4", "--epochs", "1",
            "--batch-size", "4", "--validation-fraction", "0.5",
            "--output-dir", str(output),
        ],
    )
    main()
    tokenizer = CharTokenizer.load(output / "tokenizer.json")
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    assert "z" not in tokenizer.token_to_id
    assert report["validation_source"] == "held_out_conversations"
    assert report["validation_examples"] > 0
