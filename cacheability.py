"""Mémoire de la capacité de cache d'une route, pour ne pas payer le préfixe au plein tarif.

Le mode préfixe n'a de sens que si le provider facture la relecture au tarif de cache. Certains
backends ne renvoient aucun ``prompt_tokens_details`` : rien ne prouve alors que le préfixe est servi
depuis le cache, et renvoyer 68 000 tokens au lieu de la fenêtre sérialisée du compresseur intégré
coûte plus cher que le comportement de base.

Politique : une route inconnue reste en mode préfixe, et l'observation décide. Deux mesures
consécutives sans lecture de cache, sur un prompt assez gros pour avoir été mis en cache, désactivent
la route pour le reste du processus. Une seule mesure ne suffit pas : le provider peut n'avoir encore
jamais vu ce préfixe exact. Un hit remet le compteur à zéro.
"""

from __future__ import annotations

import threading

#: En dessous, un prompt peut légitimement n'être jamais mis en cache par le provider.
MIN_PROMPT_TOKENS = 2048
#: Observations sans cache avant de désactiver la route.
MAX_STRIKES = 2
#: Routes mémorisées au plus (bornage mémoire, éviction de la plus ancienne).
MAX_ROUTES = 32

RouteKey = tuple[str, str, str]


def route_key(provider: object, model: object, base_url: object) -> RouteKey:
    return (str(provider or ""), str(model or ""), str(base_url or ""))


class RouteCacheTracker:
    """État « cette route sert-elle depuis le cache ? » par couple provider/modèle/URL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[RouteKey, int] = {}

    def prefix_allowed(self, route: RouteKey) -> bool:
        with self._lock:
            return self._strikes.get(route, 0) < MAX_STRIKES

    def record(self, route: RouteKey, prompt_tokens: object, cached_tokens: object) -> None:
        try:
            prompt = int(prompt_tokens or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if prompt < MIN_PROMPT_TOKENS:
            return
        with self._lock:
            if cached_tokens:
                self._strikes[route] = 0
                return
            self._strikes[route] = min(MAX_STRIKES, self._strikes.get(route, 0) + 1)
            while len(self._strikes) > MAX_ROUTES:
                self._strikes.pop(next(iter(self._strikes)))

    def clear(self) -> None:
        with self._lock:
            self._strikes.clear()
