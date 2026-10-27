from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import retrieve_node, grade_documents_node, transform_query_node, fallback_node, pedagogical_tutor_node, \
    feynman_eval_node


def decide_to_generate(state: AgentState) -> str:
    grader_score = state.get('grader_score')
    retry_count = state.get('retry_count', 0)

    if grader_score == "yes":
        return "tutor"
    elif grader_score == "no" and retry_count >= 2:
        return "fallback"
    else:
        return "query_transformer"


def route_entry_point(state: AgentState) -> str:
    if state.get('is_feynman_active') == True:
        return "feynman_eval"
    return "retriever"


def create_crag_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("grader", grade_documents_node)
    workflow.add_node("query_transformer", transform_query_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("tutor", pedagogical_tutor_node)
    workflow.add_node("feynman_eval", feynman_eval_node)

    workflow.add_conditional_edges(START, route_entry_point,
                                   {
                                       "feynman_eval": "feynman_eval",
                                       "retriever": "retriever"
                                   })

    workflow.add_edge("retriever", "grader")
    workflow.add_conditional_edges("grader", decide_to_generate,
                                   {
                                       "tutor": "tutor",
                                       "fallback": "fallback",
                                       "query_transformer": "query_transformer"
                                   })


    workflow.add_edge("query_transformer", "retriever")
    workflow.add_edge("tutor", END)
    workflow.add_edge("fallback", END)
    workflow.add_edge("feynman_eval", END)

    return workflow.compile()