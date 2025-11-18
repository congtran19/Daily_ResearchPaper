from abc import ABC, abstractmethod
from data.models import Paper

class BaseSource(ABC):
    @abstractmethod
    def fetch_latest(self, limit: int = 5) -> list[Paper]:
        pass
