from dotenv import load_dotenv

load_dotenv()

from engines.crag_agent.graph import create_crag_graph


def run_interactive_session():
    print("=" * 60)
    print("PEDAGÓGIAI CRAG ÁGENS - INTERAKTÍV KONZOLOS TESZT")
    print("=" * 60)

    module = input("Válassz modult ('human' vagy 'real') [alapértelmezett: human]: ").strip().lower()
    if module not in ["human", "real"]:
        module = "human"

    print(f"\n[+] Kiválasztott modul: {module.upper()}")
    print("[+] Kilépéshez írd be: 'exit' vagy 'quit'\n" + "-" * 60)

    # Gráf példányosítása
    graph = create_crag_graph()

    # Perzisztens munkamenet állapot
    state = {
        "user_query": "",
        "chat_history": [],
        "module_type": module,
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

    while True:
        try:
            user_input = input("\n🧑 Diák: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\n[!] Munkamenet lezárva.")
                break

            # Állapot frissítése az új bemenettel
            state["user_query"] = user_input

            # Gráf futtatása
            state = graph.invoke(state)

            # AI válasz kiírása
            print(f"\n🤖 Tutor: {state.get('final_response')}\n")

            # Pedagógiai telemetria panel
            print("-" * 40)
            print(f"📊 [Állapot]:")
            print(f"  • Kognitív Zóna: {state.get('cognitive_zone')}")
            print(f"  • Állványozási szint (Scaffold): {state.get('scaffold_level')}")
            print(f"  • Feynman mód aktív: {state.get('is_feynman_active')}")
            print(f"  • Feynman sikeres: {state.get('feynman_passed')}")
            print(f"  • Hibaszámláló: {state.get('consecutive_errors')}")
            print(f"  • Grader értékelés: {state.get('grader_score')}")
            print("-" * 40)

            # Chat előzmények frissítése a következő körre
            state["chat_history"].append({"role": "user", "content": user_input})
            state["chat_history"].append({"role": "assistant", "content": state.get("final_response")})

        except KeyboardInterrupt:
            print("\n[!] Kilépés...")
            break


if __name__ == "__main__":
    run_interactive_session()