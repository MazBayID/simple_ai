import json

from simple_ai.experiments.train_conversation_model import main
from simple_ai.language_model import TinyLanguageModel
from simple_ai.tokenizer import BPETokenizer, load_tokenizer


def test_bpe_training_writes_matching_model_tokenizer_and_report(tmp_path, monkeypatch):
    corpus = tmp_path / "indonesian.txt"
    corpus.write_text(
        ("Bahasa Indonesia memiliki banyak kata. Kita belajar bahasa bersama.\n" * 8),
        encoding="utf-8",
    )
    output = tmp_path / "model"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_conversation_model",
            "--input", str(corpus),
            "--tokenizer", "bpe",
            "--vocab-size", "48",
            "--context-size", "8",
            "--embedding-dim", "2",
            "--hidden-size", "8",
            "--epochs", "1",
            "--batch-size", "16",
            "--output-dir", str(output),
        ],
    )

    main()

    model = TinyLanguageModel.load(output / "model.json")
    tokenizer = load_tokenizer(output / "tokenizer.json")
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    assert isinstance(tokenizer, BPETokenizer)
    assert model.vocab_size == tokenizer.vocab_size
    assert report["tokenizer"] == "bpe"
    assert report["learned_merges"] > 0
    assert report["training_examples"] > 0
