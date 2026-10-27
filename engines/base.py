from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, Optional, Any


class TelemetryData(TypedDict):
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EngineResponse(TypedDict):
    response_text: str
    scaffold_level: Optional[int]  # 1: Pointing, 2: Teaching, 3: Bottom-Out, None: ha nem ágens
    cognitive_zone: Optional[str]  # 'ZAD', 'ZPD', 'ZDD', None: ha nem ágens
    grader_evaluation: Optional[str]  # 'correct', 'hallucination', 'fallback', None: ha nincs grader
    retry_count: int  # Hány javítási ciklus futott le
    telemetry: TelemetryData


class BaseEngine(ABC):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_response(
            self,
            user_query: str,
            chat_history: List[Dict[str, str]],
            module_type: str = "human"
    ) -> EngineResponse:
        pass