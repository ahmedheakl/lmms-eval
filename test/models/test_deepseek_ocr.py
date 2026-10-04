from types import SimpleNamespace

from lmms_eval.models import get_model
from lmms_eval.models.chat import deepseek_ocr
from lmms_eval.models.chat.deepseek_ocr import DeepSeekOCR, deepseek_ocr_to_markdown

_RAW = "<|ref|>title<|/ref|><|det|>[[80, 60, 900, 110]]<|/det|>\n# Annual report\n\n<|ref|>text<|/ref|><|det|>[[80, 140, 900, 300]]<|/det|>\nRevenue grew by 12%.\n\n<|ref|>table<|/ref|><|det|>[[80, 320, 900, 600]]<|/det|>\n<center><table><tr><td>2025</td></tr></table></center><｜end▁of▁sentence｜>"


def test_deepseek_ocr_is_registered_as_a_chat_model():
    assert get_model("deepseek_ocr") is DeepSeekOCR
    assert DeepSeekOCR.is_simple is False


def test_grounding_tags_are_removed_from_the_response():
    markdown = deepseek_ocr_to_markdown(_RAW)

    assert "<|ref|>" not in markdown and "<|det|>" not in markdown
    assert "<center>" not in markdown and "end▁of▁sentence" not in markdown
    assert "# Annual report" in markdown
    assert "Revenue grew by 12%." in markdown
    assert "<table><tr><td>2025</td></tr></table>" in markdown


def test_equation_numbers_are_dropped_from_display_formulas():
    assert deepseek_ocr_to_markdown("\\[ E = mc^2 \\quad (3) \\]") == "\\[E = mc^2\\]"


class _FakeCompletions:
    def __init__(self):
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=_RAW))])


def test_request_recipe_is_applied_to_every_request():
    inner = _FakeCompletions()
    extra_body = {"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 40, "window_size": 90, "whitelist_token_ids": [128821, 128822]}}
    client = deepseek_ocr._Client(SimpleNamespace(chat=SimpleNamespace(completions=inner), models="models"), extra_body, None)

    response = client.chat.completions.create(model="deepseek-ocr", messages=[], max_tokens=8192, temperature=0)

    payload = inner.payloads[0]
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0
    assert payload["extra_body"] == extra_body
    assert response.choices[0].message.content == deepseek_ocr_to_markdown(_RAW)
    assert client.models == "models"


def test_max_new_tokens_caps_the_request_when_set():
    inner = _FakeCompletions()
    client = deepseek_ocr._Client(SimpleNamespace(chat=SimpleNamespace(completions=inner)), {}, 4096)

    client.chat.completions.create(model="deepseek-ocr", messages=[], max_tokens=8192)
    client.chat.completions.create(model="deepseek-ocr", messages=[], max_tokens=1024)

    assert [payload["max_tokens"] for payload in inner.payloads] == [4096, 1024]
