from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseQuoteClassifier(ABC):
    @abstractmethod
    def classify(self, quotes: List[str]) -> Tuple[List[str], List[str], List[float]]:
        """
        Classifica uma lista de quotes e retorna:
        - Lista com constructos atribuídos
        - Lista com justificativas
        - Lista com tempos de execução por quote
        """
        pass