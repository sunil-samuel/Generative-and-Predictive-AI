from abc import ABC, abstractmethod

class PipelineBase(ABC):
    
    @abstractmethod
    def __init__(self, device=-1):
        pass
    
    @abstractmethod
    def process(self, prompt:str, model_name:str, context:str) -> str:
        pass
    
    @abstractmethod
    def get_model_names(self) -> list[str]:
        pass