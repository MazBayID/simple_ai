from simple_ai.storage import LocalStore


def test_training_record_and_messages(tmp_path):
    store = LocalStore(tmp_path / "simple_ai.db")
    run_id = store.record_training(
        "sample.txt",
        {
            "characters": 42,
            "vocabulary_size": 10,
            "transition_contexts": 11,
        },
    )
    assert run_id > 0

    store.add_message("user", "Hello")
    store.add_message("assistant", "Hi")
    assert store.recent_messages() == [("user", "Hello"), ("assistant", "Hi")]


def test_invalid_role_rejected(tmp_path):
    store = LocalStore(tmp_path / "simple_ai.db")
    try:
        store.add_message("alien", "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid role should be rejected")
