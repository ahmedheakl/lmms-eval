"""DeepSeek-OCR served by vLLM's OpenAI-compatible server.

DeepSeek-OCR (https://github.com/deepseek-ai/DeepSeek-OCR) is a document parsing
model with its own request recipe, which a generic OpenAI-compatible backend
cannot express:

* decoding uses vLLM's n-gram no-repeat logits processor with ``<td>`` / ``</td>``
  whitelisted, passed per request through ``vllm_xargs``;
* special tokens must be kept, because the layout of the page is returned as
  ``<|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`` tags in front of every
  block;
* prompt and output share an 8192-token context, so no ``max_tokens`` is sent and
  generation runs until the end-of-sequence token or the end of the context.

This backend adds those request parameters and converts the response to plain
Markdown exactly as the benchmark script of the DeepSeek-OCR repository does
(``run_dpsk_ocr_eval_batch.py``): grounding tags are dropped, blank lines are
collapsed, ``<center>`` tags are removed.

Start the server with the flags of the vLLM recipe::

    vllm serve deepseek-ai/DeepSeek-OCR \
        --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
        --no-enable-prefix-caching --mm-processor-cache-gb 0

and evaluate with::

    lmms-eval --model deepseek_ocr \
        --model_args model_version=deepseek-ai/DeepSeek-OCR,base_url=http://localhost:8000/v1,api_key=EMPTY \
        --tasks docatlas_bench --batch_size 1

The task has to supply the model's own instruction, e.g.
``<|grounding|>Convert the document to markdown.`` (tasks do so through
``lmms_eval_specific_kwargs`` under the ``deepseek_ocr`` key).
"""

import re
from typing import Any, Optional

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible

# Token ids of <td> and </td>: exempt from the no-repeat rule so that tables can repeat cells.
_WHITELIST_TOKEN_IDS = [128821, 128822]
_END_OF_SENTENCE = "<｜end▁of▁sentence｜>"
_GROUNDING_TAG = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
_DISPLAY_FORMULA = re.compile(r"\\\[(.*?)\\\]")


def _clean_formula(text: str) -> str:
    """Drop equation numbers such as ``\\quad (3)`` from display formulas."""

    def process_formula(match: "re.Match[str]") -> str:
        formula = re.sub(r"\\quad\s*\([^)]*\)", "", match.group(1))
        return r"\[" + formula.strip() + r"\]"

    return _DISPLAY_FORMULA.sub(process_formula, text)


def deepseek_ocr_to_markdown(text: str) -> str:
    """Convert a raw DeepSeek-OCR response (with grounding tags) to Markdown."""
    text = _clean_formula(text.replace(_END_OF_SENTENCE, ""))
    for tag in _GROUNDING_TAG.findall(text):
        text = text.replace(tag, "").replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n").replace("<center>", "").replace("</center>", "")
    return text


class _Completions:
    """``client.chat.completions`` with the DeepSeek-OCR request recipe applied."""

    def __init__(self, completions: Any, extra_body: dict, max_new_tokens: Optional[int]) -> None:
        self._completions = completions
        self._extra_body = extra_body
        self._max_new_tokens = max_new_tokens

    def create(self, **payload: Any) -> Any:
        requested = payload.pop("max_tokens", None)
        if self._max_new_tokens is not None:
            payload["max_tokens"] = self._max_new_tokens if requested is None else min(requested, self._max_new_tokens)
        payload["extra_body"] = {**(payload.get("extra_body") or {}), **self._extra_body}
        response = self._completions.create(**payload)
        for choice in response.choices:
            choice.message.content = deepseek_ocr_to_markdown(choice.message.content or "")
        return response


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, client: Any, extra_body: dict, max_new_tokens: Optional[int]) -> None:
        self._client = client
        self.chat = _Chat(_Completions(client.chat.completions, extra_body, max_new_tokens))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


@register_model("deepseek_ocr")
class DeepSeekOCR(OpenAICompatible):
    """DeepSeek-OCR behind a vLLM OpenAI-compatible server (see the module docstring)."""

    is_simple = False

    def __init__(
        self,
        *args: Any,
        ngram_size: int = 40,
        window_size: int = 90,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param ngram_size: size of the n-grams that may not repeat (40 in the model's benchmark script).
        :param window_size: number of trailing tokens the no-repeat rule looks at.
        :param max_new_tokens: optional cap on generated tokens; by default generation
            runs to the end of the model's 8192-token context.
        """
        super().__init__(*args, **kwargs)
        extra_body = {
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": int(ngram_size), "window_size": int(window_size), "whitelist_token_ids": _WHITELIST_TOKEN_IDS},
        }
        self.client = _Client(self.client, extra_body, None if max_new_tokens is None else int(max_new_tokens))
