import os
import time
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

from langchain_openai import ChatOpenAI
from .base import BaseEngine, EngineResponse, TelemetryData

class BaselineLLMEngine(BaseEngine):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(model_name=model_name, temperature=temperature)
        self.client = ChatOpenAI(
            model = self.model_name,
            temperature = self.temperature,
        )

    def generate_response(
            self,
            user_query: str,
            chat_history: List[Dict[str, str]],
            module_type: str = "human"
    ) -> EngineResponse:
        start_time = time.time()
        template = """
        Te egy egyetemi tanár vagy.

        A feladatod: válaszolj a feltett kérdés(ek)re.
        
        Szigorú szabályok:
        1. Csak magyarul válaszolj.
        2. Válaszolj röviden, tisztán, nyelvtanilag helyesen.
        
        Tanuló kérdése: {user_query}
        Eddigi beszélgetési előzmények: {chat_history}
        """
        prompt = ChatPromptTemplate.from_template(template)

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        chain = prompt | self.client

        try:
            response = chain.invoke({
                "user_query": user_query,
                "chat_history": formatted_history
            })
            response_text = response.content if isinstance(response.content, str) else str(response.content)
            latency_seconds = time.time() - start_time
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            telemetry_data = TelemetryData(latency_seconds=latency_seconds, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            return EngineResponse(response_text = response_text, scaffold_level=None, cognitive_zone=None, grader_evaluation=None, retry_count=0, telemetry=telemetry_data)

        except Exception as e:
            latency_seconds = time.time() - start_time
            telemetry_data = TelemetryData(latency_seconds=latency_seconds, prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return EngineResponse(response_text = "", scaffold_level=None, cognitive_zone=None, grader_evaluation=None, retry_count=0, telemetry=telemetry_data)