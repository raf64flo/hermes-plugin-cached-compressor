"""Mémoire de la capacité de cache d'une route, pour ne pas payer le préfixe au plein tarif.

Le mode préfixe n'a de sens que si le provider facture la relecture au tarif de cache. Certains
backends ne renvoient aucun ``prompt_tokens_details`` : rien ne prouve alors que le préfixe est servi
depuis le cache, et renvoyer 68 000 tokens au lieu de la fenêtre sérialisée du compresseur intégré
coûte plus cher que le comportement de base.

Politique : une route inconnue reste en mode préfixe, et l'observation décide. Deux mesures
consécutives sans lecture de cache, sur un prompt assez gros pour avoir été mis en cache, désactivent
la route pour le reste du processus. Une seule mesure ne suffit pas : le provider peut n'avoir encore
jamais vu ce préfixe exact. Un hit remet le compteur à zéro.

Deux leviers, tous deux optionnels et sans effet par défaut :

* un seuil de rentabilité (``set_min_ratio``) : si le taux de hit observé sur les dernières mesures
  de la route est en dessous, le mode préfixe est refusé même sans strike ;
* un repli forcé (``force_builtin``) : la route n'utilise jamais le mode préfixe, par exemple sur un
  backend connu pour ne pas mettre en cache.
"""

from __future__ import annotations

import threading

#: En dessous, un prompt peut légitimement n'être jamais mis en cache par le provider.
MIN_PROMPT_TOKENS = 2048
#: Observations sans cache avant de désactiver la route.
MAX_STRIKES = 2
#: Routes mémorisées au plus (bornage mémoire, éviction de la plus ancienne).
MAX_ROUTES = 32
#: Mesures récentes retenues par route pour le taux de hit observé.
OBSERVATION_WINDOW = 8

# Origines d'une mesure de cache. La boucle principale et la compaction ne répondent pas à la même
# question : la première dit « cette route rapporte-t-elle des lectures de cache ? », la seconde
# « sert-elle le préfixe de compaction ? ». Les mélanger rendait le seuil inopérant : une route qui
# sert 99 % à la boucle principale et 0 % aux compactions garde un taux observé au-dessus de tout
# seuil réaliste, et chaque hit de la boucle remet les strikes à zéro.
ORIGIN_MAIN = "main"
ORIGIN_COMPRESSION = "compression"

RouteKey = tuple[str, str, str]


# Régulation du processus. Un hook (``state.observe_usage``) charge ce module directement, il ne
# passe pas par ``__init__.py`` : l'état vit ici, pas dans le moteur.
_MIN_RATIO: float | None = None
_FORCED_BUILTIN = False


def set_min_ratio(value: object) -> None:
    """Seuil de rentabilité : taux de hit observé en dessous duquel le préfixe est refusé.

    ``None`` (et toute valeur illisible ou hors de ``0..1``) désactive le seuil : seul le compteur
    de strikes décide, comme par défaut.
    """
    global _MIN_RATIO
    try:
        ratio = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _MIN_RATIO = None
        return
    _MIN_RATIO = ratio if 0.0 < ratio <= 1.0 else None


def force_builtin(enabled: bool) -> None:
    """Repli forcé : la route n'utilise jamais le mode préfixe, qu'il y ait des hits ou non."""
    global _FORCED_BUILTIN
    _FORCED_BUILTIN = bool(enabled)


def is_forced_builtin() -> bool:
    return _FORCED_BUILTIN


def reset_tuning() -> None:
    """Défauts : sans seuil, sans repli forcé. Pour les tests."""
    global _MIN_RATIO, _FORCED_BUILTIN
    _MIN_RATIO = None
    _FORCED_BUILTIN = False


def route_key(provider: object, model: object, base_url: object) -> RouteKey:
    return (str(provider or ""), str(model or ""), str(base_url or ""))


class RouteCacheTracker:
    """État « cette route sert-elle depuis le cache ? » par couple provider/modèle/URL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[RouteKey, int] = {}
        self._observations: dict[RouteKey, list[tuple[int, int]]] = {}
        # Mesures de la seule compaction : c'est le taux qui décide de la rentabilité du préfixe.
        self._prefix_strikes: dict[RouteKey, int] = {}
        self._prefix_observations: dict[RouteKey, list[tuple[int, int]]] = {}

    def prefix_allowed(self, route: RouteKey) -> bool:
        with self._lock:
            if _FORCED_BUILTIN:
                return False
            if self._strikes.get(route, 0) >= MAX_STRIKES:
                return False
            if _MIN_RATIO is not None:
                # Le taux de la compaction d'abord : c'est lui qui dit si le préfixe est servi. Le
                # taux global ne sert de repli qu'avant la première compaction, sinon les hits de la
                # boucle principale (99 % sur une route qui ne sert rien à la compaction) diluent le
                # signal et aucun seuil ne peut se déclencher.
                ratio = self._ratio_unlocked(route, ORIGIN_COMPRESSION)
                if ratio is None:
                    ratio = self._ratio_unlocked(route, ORIGIN_MAIN)
                if ratio is not None and ratio < _MIN_RATIO:
                    return False
            return True

    def observed_ratio(self, route: RouteKey, origin: str | None = None) -> float | None:
        """Taux de hit sur les ``OBSERVATION_WINDOW`` dernières mesures ; ``None`` sans mesure.

        ``origin`` restreint la fenêtre aux mesures de la boucle principale (``"main"``) ou à celles
        de la compaction (``"compression"``). Sans ``origin``, toutes les mesures comptent, comme
        avant.
        """
        with self._lock:
            if origin is None:
                return self._ratio_unlocked(route, None)
            return self._ratio_unlocked(route, origin)

    def _ratio_unlocked(self, route: RouteKey, origin: str | None) -> float | None:
        if origin == ORIGIN_COMPRESSION:
            observations = self._prefix_observations.get(route)
        elif origin == ORIGIN_MAIN:
            observations = self._observations.get(route)
        else:
            observations = (self._observations.get(route) or []) + (self._prefix_observations.get(route) or [])
        if not observations:
            return None
        prompt_total = sum(prompt for prompt, _ in observations)
        if not prompt_total:
            return None
        return sum(cached for _, cached in observations) / prompt_total

    def record(self, route: RouteKey, prompt_tokens: object, cached_tokens: object,
               origin: str = ORIGIN_MAIN) -> None:
        try:
            prompt = int(prompt_tokens or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if prompt < MIN_PROMPT_TOKENS:
            return
        try:
            cached = min(int(cached_tokens or 0), prompt)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            cached = 0
        compression = origin == ORIGIN_COMPRESSION
        with self._lock:
            if cached:
                self._strikes[route] = 0
            else:
                self._strikes[route] = min(MAX_STRIKES, self._strikes.get(route, 0) + 1)
            if compression:
                # Un hit de la compaction remet son propre compteur à zéro ; un hit de la boucle
                # principale ne le remet pas, sinon deux compactions manquées ne s'additionnent
                # jamais et le préfixe continue d'être payé au plein tarif.
                self._prefix_strikes[route] = 0 if cached else min(
                    MAX_STRIKES, self._prefix_strikes.get(route, 0) + 1)
                series = self._prefix_observations.setdefault(route, [])
            else:
                series = self._observations.setdefault(route, [])
            series.append((prompt, cached))
            del series[:-OBSERVATION_WINDOW]
            while len(self._strikes) > MAX_ROUTES:
                evicted = next(iter(self._strikes))
                self._strikes.pop(evicted)
                self._observations.pop(evicted, None)
                self._prefix_strikes.pop(evicted, None)
                self._prefix_observations.pop(evicted, None)

    def prefix_strikes(self, route: RouteKey) -> int:
        """Compactions consécutives sans lecture de cache sur cette route (0..``MAX_STRIKES``)."""
        with self._lock:
            return self._prefix_strikes.get(route, 0)

    def clear(self) -> None:
        with self._lock:
            self._strikes.clear()
            self._observations.clear()
            self._prefix_strikes.clear()
            self._prefix_observations.clear()
