import json

from simple_ai.core import backend_name
from simple_ai.experiments.evaluate_generation import evaluate_prompts
from simple_ai.language_model import TinyLanguageModel
from simple_ai.tokenizer import CharTokenizer


def test_generation_evaluation_is_reproducible_and_reports_metrics():
    tokenizer = CharTokenizer.from_text("Halo dunia.")
    model = TinyLanguageModel(
        tokenizer.vocab_size, 4, embedding_dim=3, hidden_size=5, seed=12
    )
    kwargs = {
        "max_new_tokens": 6,
        "temperature": 0.8,
        "top_k": 4,
        "seed": 21,
    }
    first = evaluate_prompts(model, tokenizer, ["Halo"], **kwargs)
    second = evaluate_prompts(model, tokenizer, ["Halo"], **kwargs)

    assert first["samples"][0]["generated_text"] == second["samples"][0]["generated_text"]
    assert first["compute_backend"] == backend_name()
    assert first["prompt_count"] == 1
    assert first["generated_tokens"] <= 6
    assert 0.0 <= first["samples"][0]["unique_token_ratio"] <= 1.0
    assert "not semantic quality scores" in first["limitations"]


def test_generation_evaluation_cli_writes_json_report(tmp_path, monkeypatch):
    from simple_ai.experiments.evaluate_generation import main

    tokenizer = CharTokenizer.from_text("Halo dunia.")
    model = TinyLanguageModel(
        tokenizer.vocab_size, 4, embedding_dim=3, hidden_size=5, seed=4
    )
    model_path = tmp_path / "model.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_path = tmp_path / "reports" / "generation.json"
    model.save(model_path)
    tokenizer.save(tokenizer_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_generation",
            "--model", str(model_path),
            "--tokenizer", str(tokenizer_path),
            "--prompt", "Halo",
            "--max-new-tokens", "4",
            "--output", str(output_path),
        ],
    )
    main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["prompt_count"] == 1
    assert report["architecture"] == "TinyLanguageModel"
    assert report["model_path"] == str(model_path)
    assert len(report["samples"]) == 1
