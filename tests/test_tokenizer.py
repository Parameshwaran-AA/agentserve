from agentserve.tokenizer import Tokenizer


def test_counts_are_positive():
    t = Tokenizer()
    assert t.count_text("hello world") > 0


def test_longer_text_costs_more_tokens():
    t = Tokenizer()
    assert t.count_text("a " * 500) > t.count_text("a " * 50)


def test_message_framing_is_included():
    t = Tokenizer()
    one = t.count_messages([{"role": "user", "content": "hi"}])
    two = t.count_messages([{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hi"}])
    assert two > one


def test_multimodal_content_parts_are_counted():
    t = Tokenizer()
    msg = [{"role": "user", "content": [{"type": "text", "text": "x" * 400}]}]
    assert t.count_messages(msg) > 10


def test_empty_messages_never_return_zero():
    """Zero-token prompts would corrupt cache capacity accounting."""
    assert Tokenizer().count_messages([]) >= 1
