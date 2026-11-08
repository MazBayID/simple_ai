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
