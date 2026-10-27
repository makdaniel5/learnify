
from typing import List, Dict

import time

from engines.base import BaseEngine, EngineResponse, TelemetryData
from engines.crag_agent.graph import create_crag_graph
from engines.crag_agent.state import AgentState


class PedagogicalCRAGEngine(BaseEngine):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(model_name = model_name, temperature = temperature)
        self.graph = create_crag_graph()

    def generate_response(
            self,
            user_query: str,
            chat_history: List[Dict[str, str]],
            module_type: str = "human"
    ) -> EngineResponse:
        start_time = time.time()
        initial_state: AgentState = {
            "user_query": user_query,
            "chat_history": chat_history,
            "module_type": module_type,
            "retrieved_docs": [],
            "grader_score": None,
            "retry_count": 0,
            "cognitive_zone": None,
            "scaffold_level": None,
            "is_feynman_active": False,
            "feynman_passed": None,
            "consecutive_errors": 0,
            "final_response": "",
            "tokens_used": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

        final_state = self.graph.invoke(initial_state)

        response_text = final_state.get("final_response")
        scaffold_level = final_state.get("scaffold_level")
        cognitive_zone = final_state.get("cognitive_zone")
        grader_evaluation = final_state.get("grader_score")
        retry_count = final_state.get("retry_count")

        latency_seconds = time.time() - start_time

        usage = final_state.get("tokens_used", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        retry_count = final_state.get("retry_count", 0)

        telemetry_data = TelemetryData(latency_seconds=latency_seconds, prompt_tokens=prompt_tokens,
                                       completion_tokens=completion_tokens, total_tokens=total_tokens)
        return EngineResponse(response_text=response_text, scaffold_level=scaffold_level, cognitive_zone=cognitive_zone,
                              grader_evaluation=grader_evaluation, retry_count=retry_count, telemetry=telemetry_data)