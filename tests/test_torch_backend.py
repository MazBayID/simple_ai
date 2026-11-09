"""Tests for the optional PyTorch cloud backend."""

import pytest

torch = pytest.importorskip("torch")

from simple_ai.tokenizer import CharTokenizer
from simple_ai.torch_backend.checkpoint import load_checkpoint, save_checkpoint
from simple_ai.torch_backend.dataset import DocumentTokenDataset
from simple_ai.torch_backend.model import GPTConfig, GPTLanguageModel


def tiny_model():
    config = GPTConfig(
        vocab_size=32,
        context_size=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    return GPTLanguageModel(config)


def test_forward_loss_and_backward():
    model = tiny_model()
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_attention_does_not_use_future_tokens():
    torch.manual_seed(11)
    model = tiny_model().eval()
    first = torch.tensor([[2, 3, 4, 5, 6, 7]])
    changed_future = first.clone()
    changed_future[0, 4:] = torch.tensor([19, 20])

    with torch.inference_mode():
        original_logits, _ = model(first)
        changed_logits, _ = model(changed_future)

    torch.testing.assert_close(
        original_logits[:, :4],
        changed_logits[:, :4],
        rtol=1e-5,
        atol=1e-6,
    )


def test_generate_preserves_training_mode_and_extends_sequence():
    model = tiny_model().train()
    prompt = torch.tensor([[2, 3]])

    generated = model.generate(prompt, max_new_tokens=3, top_k=5)

    assert generated.shape == (1, 5)
    assert model.training


def test_document_dataset_never_crosses_document_boundaries():
    documents = ["abcdefgh", "ijklmnop", "qrstuvwx"]
    tokenizer = CharTokenizer.from_text("".join(documents))
    dataset = DocumentTokenDataset(
        documents, tokenizer, context_size=4
    )

    assert len(dataset) == sum(
        len(tokenizer.encode(doc, add_bos=True, add_eos=True)) - 4
        for doc in documents
    )

    allowed_character_sets = [set(doc) for doc in documents]
    for inputs, targets in dataset:
        decoded = tokenizer.decode(inputs.tolist()) + tokenizer.decode(
            targets[-1:].tolist()
        )
        characters = set(decoded)
        assert any(characters <= allowed for allowed in allowed_character_sets)


def test_checkpoint_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(23)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    _, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        expected, _ = model(inputs)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        step_in_epoch=7,
        global_step=107,
        best_validation_loss=2.5,
        config={"model": model.get_config()},
    )

    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=1e-3
    )
    state = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        restore_rng=False,
    )

    restored_model.eval()
    with torch.inference_mode():
        actual, _ = restored_model(inputs)

    torch.testing.assert_close(actual, expected)
    assert state["epoch"] == 2
    assert state["step_in_epoch"] == 7
    assert state["global_step"] == 107
    assert state["best_validation_loss"] == 2.5
