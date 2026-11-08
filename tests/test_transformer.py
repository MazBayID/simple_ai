import math

from simple_ai.dataset import NextTokenDataset
from simple_ai.language_model import load_language_model, train_language_model
from simple_ai.losses import CrossEntropyLoss
from simple_ai.optim import SGD
from simple_ai.transformer import TinyTransformerLanguageModel


def test_transformer_forward_shape_and_parameters():
    model = TinyTransformerLanguageModel(9, 4, embedding_dim=4, seed=2)
    assert model([[1, 2, 3, 4], [4, 3, 2, 1]]).shape == (2, 9)
    assert model.parameter_count() > 0
    assert len(model.parameters()) == 8


def test_transformer_backpropagates_through_attention_and_embedding():
    model = TinyTransformerLanguageModel(6, 3, embedding_dim=4, seed=3)
    loss = CrossEntropyLoss()(model([[1, 2, 3], [3, 2, 1]]), [2, 1])
    loss.backward()
    assert all(math.isfinite(value) for parameter in model.parameters()
               for row in parameter.grad for value in row)
    assert any(abs(value) > 0 for row in model.embedding.weights.grad for value in row)
    assert any(abs(value) > 0 for row in model.query.weights.grad for value in row)


def test_transformer_checkpoint_round_trip(tmp_path):
    model = TinyTransformerLanguageModel(7, 3, embedding_dim=4, seed=4)
    expected = model([[1, 2, 3]]).data
    path = tmp_path / "transformer.json"
    model.save(path)
    restored = TinyTransformerLanguageModel.load(path)
    assert restored([[1, 2, 3]]).data == expected
    assert isinstance(load_language_model(path), TinyTransformerLanguageModel)


def test_transformer_can_train_and_generate():
    ids = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    dataset = NextTokenDataset(ids, context_size=2)
    model = TinyTransformerLanguageModel(3, 2, embedding_dim=4, seed=5)
    optimizer = SGD(model.parameters(), learning_rate=0.03)
    history = train_language_model(model, dataset, optimizer, epochs=2, batch_size=2, seed=1)
    assert len(history) == 2
    generated = model.generate([1, 2], max_new_tokens=3, seed=8)
    assert len(generated) == 3
    assert all(0 <= token < 3 for token in generated)
