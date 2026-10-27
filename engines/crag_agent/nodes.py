from typing import List

from pydantic import BaseModel, Field
from engines.crag_agent.state import AgentState
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

HUMAN_DB_PATH = "./db/human"
REAL_DB_PATH = "./db/real"

def retrieve_node(state: AgentState) -> dict[str, List]:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    if state["module_type"] == "human":
        vectorstore = Chroma(persist_directory=HUMAN_DB_PATH,
                                        embedding_function=embedding)
    else:
        vectorstore = Chroma(persist_directory=REAL_DB_PATH,
                                       embedding_function=embedding)

    results = vectorstore.similarity_search(state["user_query"], k=3)

    documents = [doc.page_content for doc in results]

    return {"retrieved_docs": documents}

def grade_documents_node(state: AgentState) -> dict:
    from pydantic import BaseModel, Field

    class GradeDocuments(BaseModel):
        binary_score: str = Field(
            description="A visszakeresett dokumentum releváns-e a kérdéshez: 'yes' vagy 'no'"
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments, include_raw=True)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """Egy egyetemi minőségbiztosító bíráló (Grader) vagy.
    A feladatod eldönteni, hogy a visszakeresett tananyagrészlet releváns-e a diák kérdéséhez, és tartalmaz-e elegendő információt a megválaszolásához.

    Ha a forrás akár részben is kapcsolódik a kérdéshez és segíti a megértést, a válasz 'yes'.
    Ha a forrás teljesen irreleváns, mellébeszél vagy nem kapcsolódik a témához, a válasz 'no'."""),
        ("human", """Visszakeresett forrásrészlet:
    {document}

    Diák kérdése:
    {user_query}""")
    ])

    grader_chain = grader_prompt | structured_llm_grader

    user_query = state["user_query"]
    documents = state.get("retrieved_docs", [])
    tokens_used = state.get("tokens_used", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    if not documents:
        return {
            "grader_score": "no",
            "retrieved_docs": []
        }

    combined_docs = "\n\n".join(documents)

    result = grader_chain.invoke({
        "user_query": user_query,
        "document": combined_docs
    })

    score: GradeDocuments = result["parsed"]
    raw_message = result["raw"]
    decision = score.binary_score.strip().lower()

    if decision == "yes":
        grader_score = "yes"
        valid_documents = documents
    else:
        grader_score = "no"
        valid_documents = []

    usage = raw_message.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    updated_tokens = {
        "prompt_tokens": tokens_used["prompt_tokens"] + prompt_tokens,
        "completion_tokens": tokens_used["completion_tokens"] + completion_tokens,
        "total_tokens": tokens_used["total_tokens"] + total_tokens
    }

    return {
        "grader_score": grader_score,
        "retrieved_docs": valid_documents,
        "tokens_used": updated_tokens
    }


def transform_query_node(state: AgentState) -> dict:
    user_query = state["user_query"]
    chat_history = state.get("chat_history", [])
    current_retry = state.get("retry_count", 0)
    current_tokens = state.get("tokens_used", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]) if chat_history else "Nincsenek előzmények."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    paraphraser_prompt = ChatPromptTemplate.from_messages([
        ("system", """Szakértő keresőoptimalizáló és szemantikai kérdés-átfogalmazó vagy.
        A feladatod: vizsgáld meg a tanuló kérdését és a beszélgetési előzményeket, majd fogalmazd át a kérdést egy olyan tiszta, kulcsszavas keresőkérdéssé, amely a leghatékonyabban megtalálja a releváns tananyagot egy vektoros adatbázisban.
        
        Szigorú szabályok:
        1. Fejtsd ki a rejtett névmásokat (pl. ha a diák azt kérdezi: "És mi a képlete?", derítsd ki az előzményekből, mire gondol).
        2. Törekedj a szakkifejezések pontos használatára.
        3. KIZÁRÓLAG az újrafogalmazott keresőkifejezést add vissza, semmi mást (ne írj bevezető szöveget)!"""),
                ("human", """Beszélgetési előzmények:
        {chat_history}
        
        Tanuló eredeti kérdése:
        {user_query}
        
        Optimalizált keresőkifejezés:""")
    ])

    paraphraser_chain = paraphraser_prompt | llm

    response = paraphraser_chain.invoke({
        "chat_history": formatted_history,
        "user_query": user_query
    })

    rewritten_query = response.content if isinstance(response.content, str) else str(response.content)

    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    updated_tokens = {
        "prompt_tokens": current_tokens.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": current_tokens.get("completion_tokens", 0) + completion_tokens,
        "total_tokens": current_tokens.get("total_tokens", 0) + total_tokens
    }

    return {
        "user_query": rewritten_query.strip(),
        "retry_count": current_retry + 1,
        "tokens_used": updated_tokens
    }

def fallback_node(state: AgentState) -> dict:
    module_type = state.get("module_type", "human")

    if module_type == "real":
        msg = "A rendelkezésre álló valószínűségszámítási tananyag nem tartalmaz információt erről a témáról. Kérlek, tegyél fel a tananyaghoz kapcsolódó kérdést!"
    else:
        msg = "A rendelkezésre álló kognitív tudományi tananyag nem tartalmaz információt erről a témáról. Kérlek, tegyél fel a tananyaghoz kapcsolódó kérdést!"

    return {
        "final_response": msg,
        "grader_score": "fallback",
        "scaffold_level": None,
        "cognitive_zone": None,
        "is_feynman_active": False
    }


def pedagogical_tutor_node(state: AgentState) -> dict:
    from pydantic import BaseModel, Field
    from typing import Literal

    class TutorDecision(BaseModel):
        cognitive_zone: Literal["ZAD", "ZPD", "ZDD"] = Field(
            description="A diák észlelt zónája: ZAD (tudja), ZPD (fejlődési zóna), ZDD (túl nehéz neki/elakadt)"
        )
        scaffold_level: Literal[1, 2, 3] = Field(
            description="1: Pointing (rámutató kérdés), 2: Teaching (fogalmi tanítás + részkérdés), 3: Bottom-Out (konkrét levezetés/megoldás)"
        )
        is_feynman_active: bool = Field(
            description="True, ha a diák sikeres válasza után saját szavas, egyszerű magyarázatot kérünk tőle a mély megértéshez"
        )
        final_response: str = Field(
            description="A diáknak küldött szokratészi, rávezető válasz vagy feladat"
        )

    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    retrieved_documents = state.get("retrieved_docs", [])
    current_tokens = state.get("tokens_used", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    consecutive_errors = state.get("consecutive_errors", 0)

    if consecutive_errors == 0:
        current_scaffold_level = 1
    elif consecutive_errors == 1:
        current_scaffold_level = 2
    else:
        current_scaffold_level = 3


    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    ) if chat_history else "Nincsenek előzmények."

    combined_docs = "\n\n---\n\n".join(retrieved_documents) if retrieved_documents else "Nincs közvetlen forrás."

    tutor_prompt = ChatPromptTemplate.from_messages([
        ("system", """Te egy egyetemi szintű, szokratészi módszert alkalmazó oktatóasszisztens vagy.
        A célod: NEM a kész válasz azonnali megadása, hanem a hallgató önálló gondolkodásra és megértésre vezetése.
        
        Rendelkezésre álló tananyag (források):
        {context_text}
        
        === PEDAGÓGIAI KERETRENDSZER ===
        
        1. Kognitív Zóna Értékelése (Vigotszkij):
           - ZAD (Zone of Actual Development): A tanuló önállóan tudja a választ, magabiztos.
           - ZPD (Zone of Proximal Development): A tanuló megértési folyamatban van, segítséggel halad.
           - ZDD (Zone of Distal Development): A feladat túl távoli, a diák teljesen elakadt vagy frusztrált.
        
        2. Állványozási Szintek (VanLehn 3-Level Scaffolding):
           - Szint 1 (Pointing / Rámutatás): Csak mutass rá a logikai résre vagy tegyél fel egy irányító kérdést. NE magyarázd el a fogalmat!
           - Szint 2 (Teaching / Tanítás): Ha a diák elakadt a Szint 1 után, magyarázd el röviden a hiányzó elméleti összefüggést, majd tegyél fel egy kisebb, konkrét részfeladatot.
           - Szint 3 (Bottom-Out / Megoldás): Ha a diák ismételten nem jut előre, add meg a konkrét levezetést/megoldási lépést, és kérdezd meg, világos-e a logika.
           
        Aktuális szint: {scaffold_level}
        
        3. Feynman-Megértésellenőrzés:
           - Ha a diák helyes választ ad vagy elérte a megértést (ZAD/ZPD vége), állítsd az is_feynman_active mezőt True-ra, és kérd meg: "Fantasztikus! Most foglald össze a lényeget olyan egyszerűen, mintha egy kezdő csoporttársadnak tanítanád meg!"
        
        === SZABÁLYOK ===
        - Kizárólag a megadott forrásokból dolgozz, ne találj ki külső definíciókat!
        - Mindig barátságos, bátorító, de szigorúan szokratészi stílusban válaszolj!
        - Soha ne add meg a végeredményt az első körben!"""),
                ("human", """Beszélgetési előzmények:
        {chat_history}
        
        Tanuló aktuális kérdése / válasza:
        {user_query}""")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm_tutor = llm.with_structured_output(TutorDecision, include_raw=True)
    tutor_chain = tutor_prompt | structured_llm_tutor

    response = tutor_chain.invoke({
        "context_text": combined_docs,
        "chat_history": formatted_history,
        "user_query": user_query,
        "scaffold_level": current_scaffold_level
    })

    decision: TutorDecision = response["parsed"]
    raw_message = response["raw"]

    usage = raw_message.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    updated_tokens = {
        "prompt_tokens": current_tokens.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": current_tokens.get("completion_tokens", 0) + completion_tokens,
        "total_tokens": current_tokens.get("total_tokens", 0) + total_tokens
    }

    is_feynman = decision.is_feynman_active or decision.cognitive_zone == "ZAD"

    if is_feynman:
        next_errors = 0
        final_scaffold = None
    else:
        next_errors = consecutive_errors + 1
        final_scaffold = current_scaffold_level

    return {
        "final_response": decision.final_response,
        "scaffold_level": final_scaffold,
        "cognitive_zone": decision.cognitive_zone,
        "is_feynman_active": is_feynman,
        "consecutive_errors": next_errors,
        "tokens_used": updated_tokens
    }

def feynman_eval_node(state: AgentState) -> dict:
    class FeynmanEvaluation(BaseModel):
        is_concept_correct: bool = Field(description="Szakmailag helyes-e a tanuló magyarázata?")
        is_simple_enough: bool = Field(description="Valóban egyszerű, analógiát használó, zsargonmentes a megfogalmazás?")
        feedback_response: str = Field(description="A diáknak szánt visszajelzés (dicséret vagy javító rávezetés)")

    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    retrieved_documents = state.get("retrieved_docs", [])
    current_tokens = state.get("tokens_used", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    ) if chat_history else "Nincsenek előzmények."

    combined_docs = "\n\n---\n\n".join(retrieved_documents) if retrieved_documents else "Nincs közvetlen forrás."

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """Szigorú, de támogató egyetemi pedagógiai bíráló vagy.
        A feladatod a tanuló Feynman-típusú magyarázatának kiértékelése. A diák azt a feladatot kapta, hogy a témát úgy magyarázza el egyszerűen, mintha egy 12 évesnek vagy egy kezdőnek tanítaná.
    
        Tananyag (hiteles referencia):
        {context_text}
    
        Értékelési szempontok:
        1. is_concept_correct: Szakmailag helytálló a magyarázat? Nem tartalmaz tárgyi tévedést?
        2. is_simple_enough: Sikerült saját szavakkal, egyszerűsítve/analógiával átadnia, és elkerülte a száraz tankönyvi szövegmásolást?
    
        Ha mindkét feltétel teljesül:
        - Gratulálj, erősítsd meg a mély megértést, és zárd le a témakört!
        Ha valamelyik nem teljesül:
        - Kedvesen mutass rá a hiányzó elemre vagy félreértésre, és kérj tőle még egy próbálkozást!"""),
            ("human", """Beszélgetési előzmények:
        {chat_history}
    
        A tanuló saját szavas magyarázata:
        {user_query}""")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_evaluator = llm.with_structured_output(FeynmanEvaluation, include_raw=True)
    eval_chain = eval_prompt | structured_evaluator

    response = eval_chain.invoke({
        "context_text": combined_docs,
        "chat_history": formatted_history,
        "user_query": user_query
    })

    evaluation: FeynmanEvaluation = response["parsed"]
    raw_message = response["raw"]

    usage = raw_message.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    updated_tokens = {
        "prompt_tokens": current_tokens.get("prompt_tokens", 0) + prompt_tokens,
        "completion_tokens": current_tokens.get("completion_tokens", 0) + completion_tokens,
        "total_tokens": current_tokens.get("total_tokens", 0) + total_tokens
    }

    is_passed = evaluation.is_concept_correct and evaluation.is_simple_enough

    return {
        "final_response": evaluation.feedback_response,
        "is_feynman_active": not is_passed,
        "feynman_passed": is_passed,
        "cognitive_zone": "ZAD" if is_passed else "ZPD",
        "scaffold_level": None if is_passed else 1,
        "consecutive_errors": 0 if is_passed else 1,
        "tokens_used": updated_tokens
    }
