# ============================================================
# TERMINAL CHATBOT
# ============================================================
# Simple terminal interface that uses the
# RAG engine.
# ============================================================

from rag_engine import (
    ask_question
)

print("\nRAG Chatbot Ready")
print("Type 'exit' to quit.\n")

while True:

    question = input(
        "\nAsk a question: "
    )

    if question.lower() == "exit":
        break

    answer, sources = ask_question(
        question
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\nSources:")

    if sources:

        for source in sources:

            print(
                f"- {source}"
            )

    else:

        print(
            "None (general knowledge response)"
        )