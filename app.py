import random
import os
import streamlit as st
import time
from dotenv import load_dotenv
from pip._internal.models import target_python
from requests.packages import target
from streamlit.runtime.state import session_state

from engines.baseline_llm import BaselineLLMEngine
from engines.standard_rag import StandardRAGEngine
from engines.crag_agent.agent_engine import PedagogicalCRAGEngine

st.set_page_config(layout="wide", page_title="Learnify")
load_dotenv()

# ==============================================================================
# 1. RENDSZER- ÉS ADAT-INICIALIZÁLÁS
# ==============================================================================

TESTHELP = 50
STUDY_DEADLINE = 900/TESTHELP
TEST_DEADLINE = 600/TESTHELP
BREAK_DEADLINE = 300/TESTHELP


if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "user_query": "",
        "chat_history": [],
        "module_type": "human",
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

if "subject_log" not in st.session_state:
    st.session_state["subject_log"] = {
        'subject_id': "",
        'group': "none",
        'real_test_score': 0,
        'human_test_score': 0,
        'response_times': []
    }

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "phase" not in st.session_state:
    st.session_state["phase"] = "AUTH"

if "finished_test1" not in st.session_state:
    st.session_state["finished_test1"] = False

if "study_deadline" not in st.session_state:
    st.session_state["study_deadline"] = 0

if "break_deadline" not in st.session_state:
    st.session_state["break_deadline"] = 0

if "test_deadline" not in st.session_state:
    st.session_state["test_deadline"] = 0

if "llm_engine" not in st.session_state:
    st.session_state["llm_engine"] = BaselineLLMEngine()

if "rag_engine" not in st.session_state:
    st.session_state["rag_engine"] = StandardRAGEngine()

if "crag_enginge" not in st.session_state:
    st.session_state["crag_engine"] = PedagogicalCRAGEngine()

# ==============================================================================
# 2. AZ EGYES KÉPERNYŐK
# ==============================================================================

def render_auth_screen():
    """1. DIA: A Belépő és beállító felület"""
    st.header("🎓 Kísérleti Rendszer - Belépés")

    EXPERIMENT_PASSWORD = "tdk"

    with st.form("auth_form"):
        subject_id = st.text_input("Résztvevő kódja (pl. SUBJ_01):").strip()
        password = st.text_input("Kísérletvezetői jelszó:", type="password")

        module_choice = st.selectbox(
            "Melyik témakörrel tanuljon a diák?",
            ["human", "real"],
            format_func=lambda x: "Kognitív torzítások (Humán)" if x == "human" else "Bayes-tétel (Reál)"
        )

        submitted = st.form_submit_button("🚀 Tanulás Indítása")

        group = random.choice(["none", "llm", "rag", "crag"])

        if submitted:
            if password != EXPERIMENT_PASSWORD:
                st.error("Hibás jelszó!")
            elif not subject_id:
                st.warning("Add meg a kódot!")
            else:
                st.session_state["subject_log"]["group"] = group
                st.session_state["subject_log"]["subject_id"] = subject_id

                if group == "llm" and "llm_engine" not in st.session_state["subject_log"]:
                    st.session_state["subject_log"]["llm_engine"] = BaselineLLMEngine()

                if group == "rag" and "rag_engine" not in st.session_state["subject_log"]:
                    st.session_state["subject_log"]["rag_engine"] = StandardRAGEngine()

                # Beállítjuk a kiválasztott modult a LangGraph ágensnek
                st.session_state["agent_state"]["module_type"] = module_choice

                # ÁTLÉPTETJÜK A DIAVETÍTŐT A CHATRE:
                st.session_state["phase"] = "STUDY"
                st.rerun()


def render_study_screen():
    """2. DIA: Maga a tanulási felület (a Chat)"""
    module = st.session_state["agent_state"]["module_type"]

    if st.session_state["study_deadline"] == 0:
        st.session_state["study_deadline"] = time.time() + STUDY_DEADLINE

    universal_timer("study_deadline", "TEST")

    st.html(
        """
        <style>
            .material-box div[data-testid="stVerticalBlockBorderWrapper"] {
                height: 68vh !important;
            }
            
            .stMarkdown {
                zoom: 0.85 !important;
            }
        
            .chat-box div[data-testid="stVerticalBlockBorderWrapper"] {
                height: 60vh !important;
            }
        </style>
        """
    )

    group = st.session_state.get("subject_log", {}).get("group")

    if group == "none":
        col_material = st.container()
        col_chat = None
    else:
        col_material, col_chat = st.columns([4, 5])

    with col_material:
        st.subheader("📖 Tananyag")
        with st.container(key="material-box"):
            with st.container(height=500):
                if module == "human":
                    with open("data/human_test.md", "r", encoding="utf-8") as f:
                        st.markdown(f.read())
                else:
                    with open("data/real_test.md", "r", encoding="utf-8") as f:
                        st.markdown(f.read())

    if group == "none":
        st.info("A kontroll csoportba kerültél: kérjük, önállóan, külső segítség nélkül dolgozd fel a tananyagot.")
    else:
        with col_chat:
            match group:
                case "llm": st.subheader("💬 Egyszerű LLM")
                case "rag": st.subheader("💬 Kereséssel kiterjesztett generatív LLM")
                case "crag": st.subheader("💬 Oktató Mentor ( CRAG )")
                case "default": st.subheader("💬 Ismeretlen Mentor")
            with st.container(key="chat-box"):
                chat_container = st.container(height=425)

            with chat_container:
                for msg in st.session_state["messages"]:
                    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(msg["content"])

            prompt = st.chat_input("Tedd fel a kérdésed...")

            if prompt:
                with chat_container:
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.write(prompt)
                    st.session_state["messages"].append({"role": "user", "content": prompt})

                    with st.chat_message("assistant", avatar="🎓"):
                        with st.spinner("A mentorod gondolkodik..."):
                            st.session_state["agent_state"]["user_query"] = prompt

                            if group == "llm":
                                engine_res = st.session_state["llm_engine"].generate_response(
                                    user_query=prompt,
                                    chat_history=st.session_state["agent_state"]["chat_history"],
                                    module_type=module
                                )
                                response = engine_res["response_text"]
                                latency = engine_res["telemetry"]["latency_seconds"]
                                st.session_state["agent_state"]["tokens_used"]["prompt_tokens"] += engine_res["telemetry"]["prompt_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["completion_tokens"] += engine_res["telemetry"]["completion_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["total_tokens"] += engine_res["telemetry"]["total_tokens"]
                                st.session_state["subject_log"]["response_times"].append(latency)
                            elif group == "rag":
                                engine_res = st.session_state["rag_engine"].generate_response(
                                    user_query=prompt,
                                    chat_history=st.session_state["agent_state"]["chat_history"],
                                    module_type=module
                                )
                                response = engine_res["response_text"]
                                latency = engine_res["telemetry"]["latency_seconds"]
                                st.session_state["agent_state"]["tokens_used"]["prompt_tokens"] += engine_res["telemetry"]["prompt_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["completion_tokens"] += engine_res["telemetry"]["completion_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["total_tokens"] += engine_res["telemetry"]["total_tokens"]
                                st.session_state["subject_log"]["response_times"].append(latency)
                            elif group == "crag":
                                engine_res = st.session_state["crag_engine"].generate_response(
                                    user_query=prompt,
                                    chat_history=st.session_state["agent_state"]["chat_history"],
                                    module_type=module
                                )
                                response = engine_res["response_text"]
                                latency = engine_res["telemetry"]["latency_seconds"]
                                st.session_state["agent_state"]["tokens_used"]["prompt_tokens"] += engine_res["telemetry"]["prompt_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["completion_tokens"] += engine_res["telemetry"]["completion_tokens"]
                                st.session_state["agent_state"]["tokens_used"]["total_tokens"] += engine_res["telemetry"]["total_tokens"]
                                st.session_state["subject_log"]["response_times"].append(latency)

                            st.write(response)

                            st.session_state["messages"].append({"role": "assistant", "content": response})
                            st.session_state["agent_state"]["chat_history"].append({"role": "user", "content": prompt})
                            st.session_state["agent_state"]["chat_history"].append(
                                {"role": "assistant", "content": response})
                st.rerun()

    clicked = st.button("Tovább a tesztre ➔")
    if clicked:
        st.session_state["study_deadline"] = 0
        st.session_state["phase"] = "TEST"
        st.rerun()


def render_test_screen():
    """3. DIA: A záróteszt képernyője"""
    st.title("📝 Záróteszt")

    if st.session_state["test_deadline"] == 0:
        st.session_state["test_deadline"] = time.time() + TEST_DEADLINE

    if st.session_state["finished_test1"]:
        universal_timer("test_deadline", "FINISHED")
    else:
        universal_timer("test_deadline", "BREAK")

    module = st.session_state["agent_state"]["module_type"]

    with st.form("test_form"):
        if module == "human":
            task1 = st.radio(key="h1", label="Mi a megerősítési torzítás (confirmation bias) lényege?", options=[
                "A meglévő hiedelmeinket alátámasztó információk keresése és a ellentmondók elutasítása",
                "Az összes rendelkezésre álló tény teljesen objektív és elfogulatlan mérlegelése",
                "A múltbeli emlékek szándékos és tudatos eltorzítása",
                "A véletlenszerű mintákban összefüggések felfedezése"
            ], index=None)

            task2 = st.radio(key="h2", label="Mi jellemző az úgynevezett horgonyzási (anchoring) torzításra?", options=[
                "Az elsőként kapott (akár irreleváns) információhoz túlzottan ragaszkodunk a döntéskor",
                "A legutolsó információ befolyásolja kizárólag a döntésünket",
                "Csak a saját csoportunk véleményét fogadjuk el alapos vizsgálat nélkül",
                "A nehéz döntéseket automatikusan elhalasztjuk későbbre"
            ], index=None)

            task3 = st.radio(key="h3", label="Mit takar az elérhetőségi heurisztika (availability heuristic)?", options=[
                "Könnyen felidézhető, drasztikus példák alapján becsüljük meg egy esemény valószínűségét",
                "Csak a boltban leginkább elérhető és olcsóbb termékeket vesszük meg",
                "Kizárólag a hivatalos statisztikai adatokra hagyatkozunk",
                "A legfrissebb híreket teljesen figyelmen kívül hagyjuk"
            ], index=None)
        else:
            task1 = st.radio(key="r1", label="Mit fejez ki a Bayes-tétel a valószínűségszámításban?", options=[
                "Feltételes valószínűséget számol egy esemény bekövetkezésére egy már ismert új információ birtokában",
                "Két független esemény szorzatának az abszolút értékét",
                "Egy adatsor számtani átlagát és szórását",
                "Egy esemény bekövetkeztének esélyét bármilyen előzetes tudás nélkül"
            ], index=None)

            task2 = st.radio(key="r2", label="Mit jelent a Bayes-képletben az előzetes valószínűség (prior)?", options=[
                "Az esemény bekövetkezésének esélyét a bizonytalan új adatok vagy tesztek ismerete előtt",
                "A teszt esetleges hibaszázalékát",
                "A végső, frissített valószínűséget a mérés után",
                "A kísérlet laboratóriumi ismétléseinek számát"
            ], index=None)

            task3 = st.radio(key="r3", label="Mire használják leggyakrabban a Bayes-tétetet a gyakorlatban (pl. orvosi diagnosztika)?", options=[
                "Annak kiszámítására, hogy egy pozitív teszt esetén mekkora az esélye a tényleges betegségnek",
                "A betegség pontos okának kémiai feltárására laboratóriumban",
                "A páciens pontos életkorának és magasságának becslésére",
                "Kizárólag elméleti matematikai modellezésre, gyakorlati haszna nincs"
            ], index=None)

        submitted = st.form_submit_button("Teszt beküldése")

        answers = {1: task1, 2: task2, 3: task3}
        missing_answers = [number for number, answer in answers.items() if answer is None]

        if submitted:
            if missing_answers:
                missing_answer_string = ", ".join(map(str, missing_answers))
                st.warning(f"⚠️ Kérlek, válaszolj még a következő kérdésekre: {missing_answer_string}!")
            else:
                if module == "human":
                    correct_answers = {
                        1: "A meglévő hiedelmeinket alátámasztó információk keresése és a ellentmondók elutasítása",
                        2: "Az elsőként kapott (akár irreleváns) információhoz túlzottan ragaszkodunk a döntéskor",
                        3: "Könnyen felidézhető, drasztikus példák alapján becsüljük meg egy esemény valószínűségét"
                    }
                    score = sum(1 for k, v in correct_answers.items() if answers[k] == v)
                    st.session_state["subject_log"]["human_test_score"] = score
                else:
                    correct_answers = {
                        1: "Feltételes valószínűséget számol egy esemény bekövetkezésére egy már ismert új információ birtokában",
                        2: "Az esemény bekövetkezésének esélyét a bizonytalan új adatok vagy tesztek ismerete előtt",
                        3: "Annak kiszámítására, hogy egy pozitív teszt esetén mekkora az esélye a tényleges betegségnek"
                    }
                    score = sum(1 for k, v in correct_answers.items() if answers[k] == v)
                    st.session_state["subject_log"]["real_test_score"] = score

                st.session_state["test_deadline"] = 0
                if st.session_state["finished_test1"]:
                    st.session_state["phase"] = "FINISHED"
                    st.rerun()
                else:
                    st.session_state["phase"] = "BREAK"
                    st.rerun()

def render_break_screen():
    _, col_center, _ = st.columns([1, 2, 1])

    with col_center:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Élvezd a szünetet! ☕</h1>", unsafe_allow_html=True)

        if st.session_state["break_deadline"] == 0:
            st.session_state["break_deadline"] = time.time() + BREAK_DEADLINE

        def break_cleanup():
            st.session_state["messages"] = []
            st.session_state["agent_state"]["chat_history"] = []
            st.session_state["finished_test1"] = True
            if st.session_state["agent_state"]["module_type"] == "human":
                st.session_state["agent_state"]["module_type"] = "real"
            else:
                st.session_state["agent_state"]["module_type"] = "human"

        universal_timer(deadline_key="break_deadline", target_phase="STUDY", on_expire=break_cleanup, style="hero")

def render_finished_screen():
    st.title("Köszönöm részvételed! 😊", text_alignment="center")

    import csv

    data = [{
        'subject_id': st.session_state.get("subject_log", {}).get("subject_id", ""),
        'group': st.session_state.get("subject_log", {}).get("group", ""),
        'real_test_score': st.session_state.get("subject_log", {}).get("real_test_score", 0),
        'human_test_score': st.session_state.get("subject_log", {}).get("human_test_score", 0),
        'chat_history': st.session_state.get("agent_state", {}).get("chat_history", []),
        'response_times': st.session_state.get("subject_log", {}).get("response_times", []),
        'tokens_used': st.session_state.get("agent_state", {}).get("tokens_used", {}).get("total_tokens", 0)
    }]

    file_exists = os.path.isfile('research_records.csv')

    with open('research_records.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subject_id', 'group', 'real_test_score', 'human_test_score', "chat_history",
                      "response_times", "tokens_used"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

    st.success("Az adataid sikeresen rögzítésre kerültek.")


@st.fragment(run_every=1)
def universal_timer(deadline_key: str, target_phase: str, on_expire=None, style=None):
    deadline = st.session_state.get(deadline_key, 0)

    if deadline == 0:
        return

    remaining = int(deadline - time.time())

    if remaining > 0:
        mm, ss = remaining // 60, remaining % 60
        if style == "hero":
            st.markdown(
                f"""
                        <div style="text-align: center; margin: 30px auto; max-width: 420px; padding: 25px 20px;
                                    border-radius: 20px; border: 1px solid #b2b2b2;
                                    box-shadow: 0 12px 30px #b2b2b2;">
                            <div style="font-size: 0.85rem; font-weight: 600; 
                                        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">
                                ⏳ Hátralévő szünet
                            </div>
                            <div style="font-size: 5rem; font-weight: 800; font-family: 'Courier New', monospace; 
                                        line-height: 1;">
                                {mm:02d}:{ss:02d}
                            </div>
                        </div>
                        """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                        <div style="position: fixed; top: 70px; right: 30px; z-index: 9999;">
                            <div style="padding: 6px 14px; border: 1px solid #313746;
                                        border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);">
                                <span style="font-size: 0.8rem; margin-right: 6px;">⏳ Idő:</span>
                                <span style="font-size: 1.25rem; font-weight: 700; font-family: monospace;">
                                    {mm:02d}:{ss:02d}
                                </span>
                            </div>
                        </div>
                        """,
                unsafe_allow_html=True
            )
    else:
        st.session_state[deadline_key] = 0

        if on_expire:
            on_expire()

        st.session_state["phase"] = target_phase
        st.rerun()

# ==============================================================================
# 3. A FŐ VEZÉRLŐ (ROUTER)
# ==============================================================================

if st.session_state["phase"] == "AUTH":
    render_auth_screen()
elif st.session_state["phase"] == "STUDY":
    render_study_screen()
elif st.session_state["phase"] == "TEST":
    render_test_screen()
elif st.session_state["phase"] == "BREAK":
    render_break_screen()
elif st.session_state["phase"] == "FINISHED":
    render_finished_screen()