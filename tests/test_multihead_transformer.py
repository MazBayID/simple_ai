import json
import math

from simple_ai.dataset import NextTokenDataset
from simple_ai.language_model import load_language_model, train_language_model
from simple_ai.losses import CrossEntropyLoss
from simple_ai.multihead_transformer import MultiHeadTransformerLanguageModel
from simple_ai.optim import Adam


def test_multihead_transformer_forward_shape_and_head_count():
    model = MultiHeadTransformerLanguageModel(11, 4, embedding_dim=8, num_heads=2, seed=2)
    assert model([[1, 2, 3, 4], [4, 3, 2, 1]]).shape == (2, 11)
    assert model.attention.num_heads == 2
    assert len(model.attention.heads) == 2
    assert model.parameter_count() > 0


def test_multihead_transformer_backpropagates_through_all_major_parts():
    model = MultiHeadTransformerLanguageModel(8, 3, embedding_dim=4, num_heads=2, seed=3)
    loss = CrossEntropyLoss()(model([[1, 2, 3], [3, 2, 1]]), [2, 1])
    loss.backward()
    assert all(math.isfinite(value) for parameter in model.parameters()
               for row in parameter.grad for value in row)
    assert any(abs(value) > 0 for row in model.embedding.weights.grad for value in row)
    assert any(abs(value) > 0 for row in model.attention.heads[0].query.weights.grad for value in row)
    assert any(abs(value) > 0 for row in model.attention_norm.weight.grad for value in row)


def test_multihead_transformer_rejects_incompatible_head_dimensions():
    import pytest
    with pytest.raises(ValueError):
        MultiHeadTransformerLanguageModel(8, 4, embedding_dim=5, num_heads=2)


def test_multihead_transformer_checkpoint_round_trip(tmp_path):
    model = MultiHeadTransformerLanguageModel(7, 3, embedding_dim=4, num_heads=2, seed=4)
    expected = model([[1, 2, 3]]).data
    path = tmp_path / "multihead.json"
    model.save(path)
    restored = MultiHeadTransformerLanguageModel.load(path)
    assert restored([[1, 2, 3]]).data == expected
    assert isinstance(load_language_model(path), MultiHeadTransformerLanguageModel)


def test_multihead_transformer_can_train_and_generate():
    ids = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    dataset = NextTokenDataset(ids, context_size=2)
    model = MultiHeadTransformerLanguageModel(3, 2, embedding_dim=4, num_heads=2, seed=5)
    optimizer = Adam(model.parameters(), learning_rate=0.003)
    history = train_language_model(model, dataset, optimizer, epochs=2, batch_size=2, seed=1)
    assert len(history) == 2
    generated = model.generate([1, 2], max_new_tokens=3, seed=8)
    assert len(generated) == 3
    assert all(0 <= token < 3 for token in generated)


def test_transformer_v2_training_cli_saves_loadable_checkpoint(tmp_path, monkeypatch):
    from simple_ai.experiments.train_conversation_model import main
    from simple_ai.tokenizer import load_tokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Belajar bahasa Indonesia bersama.\n" * 6), encoding="utf-8")
    output = tmp_path / "transformer-v2-run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_conversation_model", "--input", str(corpus),
            "--tokenizer", "bpe", "--architecture", "transformer_v2",
            "--vocab-size", "32", "--context-size", "4",
            "--embedding-dim", "4", "--num-heads", "2",
            "--epochs", "1", "--batch-size", "8",
            "--output-dir", str(output),
        ],
    )
    main()
    model = load_language_model(output / "model.json")
    tokenizer = load_tokenizer(output / "tokenizer.json")
    assert isinstance(model, MultiHeadTransformerLanguageModel)
    assert model.vocab_size == tokenizer.vocab_size
    report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    assert report["optimizer"] == "adam"
    assert report["attention_heads"] == 2
