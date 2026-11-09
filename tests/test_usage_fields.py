"""Lecture du cache dans l'usage, selon le backend.

Chaque famille de provider nomme autrement les tokens relus depuis son cache. Le plugin doit les
reconnaître tous, sinon il conclut « pas de cache » sur un backend qui en a un, et la garde de
cacheabilité le renvoie au compresseur intégré sans raison.
"""

from __future__ import annotations

import pytest
from support import load_plugin

_plugin = load_plugin()
_cached_tokens = _plugin._cached_tokens  # type: ignore[attr-defined]


class _Obj:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def _response(usage: object) -> _Obj:
    return _Obj(usage=usage)


@pytest.mark.parametrize("usage", [
    # OpenAI, vLLM, routeurs compatibles : le champ est imbriqué.
    _Obj(prompt_tokens_details=_Obj(cached_tokens=1_024)),
    _Obj(prompt_tokens_details={"cached_tokens": 1_024}),
    # DeepSeek : à la racine.
    {"prompt_cache_hit_tokens": 1_024},
    # Anthropic.
    {"cache_read_input_tokens": 1_024},
    # Gemini, en snake_case ou en camelCase.
    _Obj(cached_content_token_count=1_024),
    {"cachedContentTokenCount": 1_024},
])
def test_variantes_connues(usage):
    assert _cached_tokens(_response(usage)) == 1_024


@pytest.mark.parametrize("usage", [None, {}, _Obj(), {"prompt_tokens": 900},
                                   {"prompt_tokens_details": {"cached_tokens": "n/a"}}])
def test_absence_de_cache_vaut_zero(usage):
    assert _cached_tokens(_response(usage)) == 0


def test_reponse_sans_usage():
    assert _cached_tokens(_Obj()) == 0
