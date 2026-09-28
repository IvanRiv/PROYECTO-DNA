from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, random_state=42):
        self.random_state = random_state

    @abstractmethod
    def create_pipeline(self):
        """
        Crea y devuelve el Pipeline del modelo.
        """
        pass

    @abstractmethod
    def get_param_grid(self):
        """
        Devuelve el espacio de hiperparámetros.
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        Devuelve el nombre del modelo.
        """
        pass
