import json
import sys

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from simple_ai.experiments import train_torch_v2


def write_jsonl(path, title, text):
    path.write_text(
        json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_v2_training_smoke_creates_resumable_artifacts(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    output = tmp_path / "run"

    write_jsonl(
        train,
        "Artikel Latihan",
        ("Bahasa Indonesia adalah bahasa yang digunakan untuk komunikasi. " * 8),
    )
    write_jsonl(
        validation,
        "Artikel Validasi",
        ("Bahasa Indonesia memiliki banyak penutur di berbagai wilayah. " * 5),
    )
    write_jsonl(
        test,
        "Artikel Uji",
        ("Teknologi membantu manusia mengolah informasi dan berkomunikasi. " * 5),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_torch_v2",
            "--train", str(train),
            "--validation", str(validation),
            "--test", str(test),
            "--tokenizer", "byte-bpe",
            "--vocab-size", "512",
            "--context-size", "8",
            "--embedding-dim", "16",
            "--num-heads", "4",
            "--num-layers", "1",
            "--batch-size", "2",
            "--epochs", "1",
            "--warmup-steps", "0",
            "--checkpoint-steps", "1",
            "--eval-batches", "1",
            "--full-final-eval",
            "--max-steps", "1",
            "--device", "cpu",
            "--output-dir", str(output),
        ],
    )

    train_torch_v2.main()

    assert (output / "tokenizer.json").is_file()
    assert (output / "checkpoint_latest.pt").is_file()
    assert (output / "checkpoint_best.pt").is_file()
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    assert report["version"] == 2
    assert report["global_step"] == 1
    assert report["final_validation_loss"] is not None
    assert report["final_test_loss"] is not None
