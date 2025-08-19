import gymnasium as gym

# Welche Action-Spaces sind erlaubt?
COMPAT = {
    "PPO":  (gym.spaces.Discrete, gym.spaces.Box),
    "A2C":  (gym.spaces.Discrete, gym.spaces.Box),
    "DQN":  (gym.spaces.Discrete,),
    "SAC":  (gym.spaces.Box,),
    "TD3":  (gym.spaces.Box,),
    "DDPG": (gym.spaces.Box,),
    "RANDOM": (gym.spaces.Discrete, gym.spaces.Box),
}

def check_compat(valid_algos: list[str], action_space: gym.spaces.Space) -> list[str]:
    """Filtert inkompatible Algos. Sonderfall: Wenn genau 1 Algo angegeben
    und inkompatibel ist -> sofort Fehler. Wenn mehrere angegeben sind,
    inkompatible verwerfen; falls danach nichts übrig bleibt -> Fehler."""
    if not valid_algos:
        raise SystemExit("Es wurden keine gültigen Algorithmen angegeben.")

    # Sonderfall: exakt ein Algo angegeben
    if len(valid_algos) == 1:
        name = valid_algos[0]
        if isinstance(action_space, COMPAT[name]):
            return valid_algos
        # direkte, klare Fehlermeldung
        raise SystemExit(
            f"Algorithmus {name} ist inkompatibel mit Action-Space "
            f"{type(action_space).__name__}.\n"
            f"Erlaubte Action-Spaces für {name}: "
            f"{', '.join(t.__name__ for t in COMPAT[name])}."
        )

    # Mehrere: inkompatible rausfiltern, warnen, aber weiter machen
    compatible = [n for n in valid_algos if isinstance(action_space, COMPAT[n])]
    if not compatible:
        # nichts übrig -> Fehler
        raise SystemExit(
            "Keine der angegebenen Algorithmen ist kompatibel mit "
            f"{type(action_space).__name__}.\n"
            "Bitte die Auswahl oder die Umgebung anpassen."
        )
    skipped = [n for n in valid_algos if n not in compatible]
    if skipped:
        print(
            "Hinweis: Überspringe inkompatible Algorithmen für "
            f"{type(action_space).__name__}: {', '.join(skipped)}"
        )
    return compatible
