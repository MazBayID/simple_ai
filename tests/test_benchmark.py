from simple_ai.experiments.benchmark_model import main
from simple_ai.language_model import TinyLanguageModel
from simple_ai.tokenizer import BPETokenizer


def test_benchmark_reports_generation_and_memory_metrics(tmp_path, monkeypatch, capsys):
    tokenizer = BPETokenizer.train("halo dunia halo dunia", vocab_size=24)
    model = TinyLanguageModel(tokenizer.vocab_size, 4, embedding_dim=2, hidden_size=4)
    model_path = tmp_path / "model.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    model.save(model_path)
    tokenizer.save(tokenizer_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_model", "--model", str(model_path),
            "--tokenizer", str(tokenizer_path), "--generation-tokens", "2",
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "tokens_per_second:" in output
    assert "python_traced_peak_mib:" in output
