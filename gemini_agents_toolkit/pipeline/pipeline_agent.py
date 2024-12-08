from abc import ABC, abstractmethod

from vertexai.generative_models import (
    GenerationConfig,
)


# Define an abstract class
class AbstractPipelineAgent(ABC):

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    @abstractmethod
    def send_message(self, msg: str, *, generation_config: GenerationConfig = None, history = None) -> tuple[str, list]:
        """This method must be overridden by subclasses"""
        pass
