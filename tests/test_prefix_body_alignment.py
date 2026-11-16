"""Le corps réellement envoyé est-il le payload capturé, octet pour octet ?

Le mode préfixe ne peut lire le cache que si les octets envoyés sont ceux du tour principal. Ce test
le vérifie sans réseau et sans clé : un faux serveur OpenAI-wire sur la boucle locale enregistre le
corps JSON que le client auxiliaire sérialise, et on le compare aux messages capturés par
``pre_api_request``.

C'est la réponse, en local, à l'hypothèse « le payload du résumé n'est pas aligné sur celui qui est
en cache ». Elle est fausse pour ce que le plugin contrôle : les messages passent tels quels, jusqu'à
l'instruction de synthèse. La seule divergence que le client introduit est le dédoublonnage des noms
d'outils (``_build_call_kwargs``), et elle est inerte ici : aucun doublon dans les journaux de
production (``duplicate tool name`` : 0 occurrence sur les quatre fichiers ``agent.log*``).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Un venv de test étranger à l'installation Hermes déclenche sa synchronisation de dépendances, qui
# relance l'interpréteur et perd pytest. Le drapeau coupe cette relance, sans effet en CI.
os.environ.setdefault("HERMES_DISABLE_LAZY_INSTALLS", "1")

import pytest

try:  # un environnement de test qui melange deux pydantic (ABI differente) ne peut pas ce test
    import openai  # noqa: F401
except Exception as exc:  # pragma: no cover - depend de la machine, pas du code
    pytest.skip(
        f"client OpenAI indisponible dans cet interpreteur ({exc}); lancer ce module avec "
        f"l'interpreteur de l'installation Hermes, ou dans le venv du depot: "
        f"HERMES_SOURCE=~/.hermes/hermes-agent pytest tests/test_prefix_body_alignment.py",
        allow_module_level=True,
    )

from support import load_plugin

_plugin = load_plugin()

REQUETE: list[dict] = []


class _Serveur(BaseHTTPRequestHandler):
    """Faux backend OpenAI-wire : enregistre le corps, rend un usage avec cache."""

    def do_POST(self):  # noqa: N802 - nom impose par BaseHTTPRequestHandler
        longueur = int(self.headers.get("content-length") or 0)
        corps = json.loads(self.rfile.read(longueur) or b"{}")
        REQUETE.append({"corps": corps, "entetes": dict(self.headers)})
        reponse = {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": corps.get("model") or "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "resume"}}],
            "usage": {"prompt_tokens": 4096, "completion_tokens": 4, "total_tokens": 4100,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }
        charge = json.dumps(reponse).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(charge)))
        self.end_headers()
        self.wfile.write(charge)

    def log_message(self, *args):  # silence: le test ne parle pas reseau
        return


@pytest.fixture()
def serveur():
    REQUETE.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Serveur)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _fil() -> dict:
    messages = [
        {"role": "system", "content": "SYSTEME. " * 200},
        {"role": "user", "content": "Premier tour. " * 40},
        {"role": "assistant", "content": "Reponse. " * 30},
        {"role": "tool", "tool_call_id": "appel-1", "content": "sortie d'outil " * 40},
        {"role": "user", "content": "Question courante."},
    ]
    outils = [{"type": "function", "function": {"name": "terminal", "description": "run",
                                                "parameters": {"type": "object", "properties": {}}}},
              {"type": "function", "function": {"name": "read_file", "description": "read",
                                                "parameters": {"type": "object", "properties": {}}}}]
    return {"messages": messages, "tools": outils}


def _resume_envoye(fil: dict) -> list[dict]:
    """Forme exacte produite par ``summary.reader_messages``: prefixe tronque + instruction."""
    return list(fil["messages"][:-1]) + [{"role": "user", "content": "Resume la conversation."}]


def _appeler(base_url: str, messages: list[dict], outils: list) -> None:
    _plugin.call_llm(  # type: ignore[attr-defined]
        task="compression", messages=messages, tools=outils,
        provider="custom", model="modele-de-test", base_url=base_url, api_key="cle-de-test",
        main_runtime={"model": "modele-de-test", "provider": "custom", "base_url": base_url,
                      "api_mode": "chat_completions", "api_key": "cle-de-test",
                      "session_id": "session-de-test"},
    )


def test_les_messages_envoyes_sont_les_messages_captures(serveur) -> None:
    fil = _fil()
    envoyes = _resume_envoye(fil)
    _appeler(serveur, envoyes, fil["tools"])

    assert len(REQUETE) == 1, "le client auxiliaire doit emettre exactement une requete"
    corps = REQUETE[0]["corps"]
    assert corps["messages"] == envoyes, "le client a reecrit les messages du prefixe"
    # La tete est identique jusqu'au dernier message du fil, y compris le message systeme.
    assert corps["messages"][:len(fil["messages"]) - 1] == fil["messages"][:-1]
    assert corps["messages"][-1] == {"role": "user", "content": "Resume la conversation."}


def test_les_outils_partent_identiques_quand_ils_sont_uniques(serveur) -> None:
    fil = _fil()
    _appeler(serveur, _resume_envoye(fil), fil["tools"])
    corps = REQUETE[0]["corps"]
    assert corps.get("tools") == fil["tools"], "les declarations d'outils doivent partir telles quelles"


def test_un_doublon_d_outil_est_la_seule_divergence_connue(serveur) -> None:
    """Le client retire les noms d'outils en double avant l'envoi : les octets divergent alors.

    Le cas est mesurable et surveillé (``tools=déclarés/uniques`` dans la ligne de diagnostic) ;
    il ne se produit pas dans les journaux de production actuels.
    """
    fil = _fil()
    doublon = dict(fil["tools"][0])
    fil["tools"] = fil["tools"] + [doublon]
    _appeler(serveur, _resume_envoye(fil), fil["tools"])
    corps = REQUETE[0]["corps"]
    assert len(corps["tools"]) == len(fil["tools"]) - 1


def test_le_corps_ne_porte_ni_cle_ni_contenu_de_session(serveur) -> None:
    """Le payload envoyé ne doit pas embarquer la clé d'API (elle part en en-tête)."""
    fil = _fil()
    _appeler(serveur, _resume_envoye(fil), fil["tools"])
    corps = json.dumps(REQUETE[0]["corps"])
    assert "cle-de-test" not in corps
    assert "session-de-test" not in corps
