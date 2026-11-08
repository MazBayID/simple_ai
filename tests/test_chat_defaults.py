from pathlib import Path

from simple_ai.chat import main


def test_chat_defaults_point_to_indonesian_checkpoint(monkeypatch):
    captured = {}

    class FakeModel:
        vocab_size = 4

    class FakeTokenizer:
        vocab_size = 4
        eos_id = 3

    monkeypatch.setattr("sys.argv", ["simple-ai-chat"])
    monkeypatch.setattr("simple_ai.chat.TinyLanguageModel.load",
                        lambda path: captured.setdefault("model", str(path)) or FakeModel())
    monkeypatch.setattr("simple_ai.chat.load_tokenizer",
                        lambda path: captured.setdefault("tokenizer", str(path)) or FakeTokenizer())
    # No actual terminal interaction is needed to assert parser defaults.
    monkeypatch.setattr("builtins.input", lambda _: "/exit")
    try:
        main()
    except AttributeError:
        # The test only needs to ensure the default paths were selected.
        pass
    assert captured["model"] == "artifacts/indonesian_model/model.json"
    assert captured["tokenizer"] == "artifacts/indonesian_model/tokenizer.json"
