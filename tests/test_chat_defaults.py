from pathlib import Path

from simple_ai.chat import main


def test_chat_defaults_point_to_indonesian_checkpoint(monkeypatch):
    captured = {}

    class FakeModel:
        vocab_size = 4

    class FakeTokenizer:
        vocab_size = 4
        eos_id = 3

    def load_model(path):
        captured["model"] = str(path)
        return FakeModel()

    def load_tokenizer(path):
        captured["tokenizer"] = str(path)
        return FakeTokenizer()

    monkeypatch.setattr("sys.argv", ["simple-ai-chat"])
    monkeypatch.setattr("simple_ai.chat.TinyLanguageModel.load", load_model)
    monkeypatch.setattr("simple_ai.chat.load_tokenizer", load_tokenizer)
    monkeypatch.setattr("builtins.input", lambda _: "/exit")
    main()

    assert captured["model"] == "artifacts/indonesian_model/model.json"
    assert captured["tokenizer"] == "artifacts/indonesian_model/tokenizer.json"
