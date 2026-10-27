from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    user_query: str
    chat_history: List[Dict[str, str]]
    module_type: str
    retrieved_docs: List[str]
    grader_score: Optional[str]
    retry_count: int
    cognitive_zone: Optional[str]
    scaffold_level: Optional[int]
    is_feynman_active: bool
    final_response: str
    tokens_used: Dict[str, int]
    consecutive_errors: int
    feynman_passed: Optional[bool]