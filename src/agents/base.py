"""
Contrat commun à tous les agents de collecte.

Un agent = une responsabilité.
Chaque agent peut être développé, testé et remplacé indépendamment.
Tous produisent le même objet VideoSnapshot — le stockage et le moteur
de viralité n'ont jamais besoin de connaître l'agent source.
"""

from abc import ABC, abstractmethod

from src.models import VideoSnapshot


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant lisible de l'agent (ex : 'KeywordAgent')."""
        ...

    @abstractmethod
    def collect(self) -> list[VideoSnapshot]:
        """
        Collecte des vidéos et retourne une liste de VideoSnapshot.
        Toute la configuration doit être fournie au constructeur.
        """
        ...
