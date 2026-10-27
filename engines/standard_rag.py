import os
import time
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

from .base import BaseEngine, EngineResponse, TelemetryData

load_dotenv()

DATA_PATH = "./data/"
HUMAN_DB_PATH = "./db/human"
REAL_DB_PATH = "./db/real"


class StandardRAGEngine(BaseEngine):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        super().__init__(model_name=model_name, temperature=temperature)
        self.client = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        # Chroma adatbázisok betöltése a memóriazárolási hiba elkerülésével
        self.human_vectorstore = Chroma(
            persist_directory=HUMAN_DB_PATH,
            embedding_function=self.embedding,
            client_settings=Settings(is_persistent=True, allow_reset=True)
        )
        self.real_vectorstore = Chroma(
            persist_directory=REAL_DB_PATH,
            embedding_function=self.embedding,
            client_settings=Settings(is_persistent=True, allow_reset=True)
        )

    def generate_response(
            self,
            user_query: str,
            chat_history: List[Dict[str, str]],
            module_type: str = "human"
    ) -> EngineResponse:
        start_time = time.time()

        if module_type == "real":
            vectorstore = self.real_vectorstore
            role_description = "Te egy egyetemi valószínűségszámítás és matematika oktatóasszisztens vagy."
        else:
            vectorstore = self.human_vectorstore
            role_description = "Te egy egyetemi pszichológia és kognitív tudományi oktatóasszisztens vagy."

        template = """
        {role_description}

        A feladatod: kizárólag a megadott forrásrészletek alapján válaszolj.
        Források:
        {context_text}

        Szigorú szabályok:
        1. Csak magyarul válaszolj.
        2. Válaszolj röviden, tisztán, nyelvtanilag helyesen.

        Tanuló kérdése: {user_query}
        Eddigi beszélgetési előzmények: {chat_history}
        """

        results = vectorstore.similarity_search(user_query, k=3)
        if not results:
            latency_seconds = time.time() - start_time
            return EngineResponse(
                response_text="A rendelkezésre álló tananyag nem tartalmaz elegendő információt a kérdés megválaszolásához.",
                scaffold_level=None,
                cognitive_zone=None,
                grader_evaluation=None,
                retry_count=0,
                telemetry=TelemetryData(latency_seconds=latency_seconds, prompt_tokens=0, completion_tokens=0,
                                        total_tokens=0)
            )

        context_text = "\n\n---\n\n".join(doc.page_content for doc in results)
        prompt = ChatPromptTemplate.from_template(template)

        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        chain = prompt | self.client

        try:
            response = chain.invoke({
                "role_description": role_description,
                "user_query": user_query,
                "chat_history": formatted_history,
                "context_text": context_text
            })
            response_text = response.content if isinstance(response.content, str) else str(response.content)
            latency_seconds = time.time() - start_time
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            telemetry_data = TelemetryData(
                latency_seconds=latency_seconds,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            return EngineResponse(
                response_text=response_text,
                scaffold_level=None,
                cognitive_zone=None,
                grader_evaluation=None,
                retry_count=0,
                telemetry=telemetry_data
            )

        except Exception as e:
            latency_seconds = time.time() - start_time
            telemetry_data = TelemetryData(
                latency_seconds=latency_seconds,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
            return EngineResponse(
                response_text=f"A rendszerben hiba lépett fel: {str(e)}",
                scaffold_level=None,
                cognitive_zone=None,
                grader_evaluation=None,
                retry_count=0,
                telemetry=telemetry_data
            )