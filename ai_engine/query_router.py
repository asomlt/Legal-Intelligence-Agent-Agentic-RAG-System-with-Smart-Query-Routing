import re

from llm_engine import generate_simple_response

from legal_retriever import run_legal_pipeline
from general_chat import run_general_chat
from unsafe_handler import run_unsafe_handler


print("\nLoading Intelligent Query Router...\n")

print("\nAI Query Router Ready 🔥\n")


# =========================================================
# ROUTING PROMPT
# =========================================================

ROUTING_PROMPT = """

You are an advanced AI Routing and Decision Engine.

Your ONLY task is to classify the user query
into ONE route.

--------------------------------------------------
AVAILABLE ROUTES
--------------------------------------------------

1. LEGAL_RAG

Choose LEGAL_RAG if the query is related to:
- laws
- IPC
- CRPC
- FIR
- police
- arrest
- warrant
- evidence
- courts
- judges
- constitution
- legal rights
- punishments
- criminal law
- civil law
- cyber law
- family law
- property disputes
- land disputes
- crimes
- bail
- legal procedures
- Indian legal system
- legal conflicts
- legal advice

2. UNSAFE

Choose UNSAFE if the query asks for:
- bomb making
- hacking
- murder instructions
- terrorism
- violence instructions
- illegal dangerous activities

3. GREETING

Choose GREETING if the query is:
- hello
- hi
- hey
- greetings
- casual welcome messages

4. GENERAL

Choose GENERAL ONLY if the query clearly
does NOT belong to:
- legal
- unsafe
- greeting

--------------------------------------------------
VERY IMPORTANT RULE
--------------------------------------------------

If the query has EVEN SLIGHT relation to:
- law
- legal issue
- crime
- police
- warrant
- FIR
- arrest
- disputes
- punishments
- evidence
- court
- rights
- constitution

Then ALWAYS classify as:

LEGAL_RAG

--------------------------------------------------
STRICT OUTPUT RULES
--------------------------------------------------

Return ONLY ONE WORD.

Allowed outputs ONLY:

LEGAL_RAG
UNSAFE
GREETING
GENERAL

Do NOT explain.
Do NOT generate sentences.
Do NOT add punctuation.
Do NOT add markdown.

--------------------------------------------------
EXAMPLES
--------------------------------------------------

Query:
What are criminal laws in India?

Output:
LEGAL_RAG

Query:
Can police arrest without evidence?

Output:
LEGAL_RAG

Query:
Land dispute with neighbour.

Output:
LEGAL_RAG

Query:
How to make bomb?

Output:
UNSAFE

Query:
Hello brother

Output:
GREETING

Query:
Latest cricket news

Output:
GENERAL

"""


# =========================================================
# QUERY CLASSIFICATION
# =========================================================

def classify_query(user_query):

    print("\nUnderstanding Query Intent...\n")

    final_prompt = f"""

{ROUTING_PROMPT}

--------------------------------------------------
USER QUERY
--------------------------------------------------

{user_query}

--------------------------------------------------
TASK
--------------------------------------------------

Classify the query into ONE route.

"""

    print("\nSending Query To Gemini Router...\n")

    response = generate_simple_response(
        final_prompt
    )

    print(f"\nRAW GEMINI RESPONSE:\n{response}\n")

    response = response.upper()

    # extract valid route
    match = re.search(

        r"(LEGAL_RAG|UNSAFE|GREETING|GENERAL)",

        response
    )

    if match:

        cleaned_response = match.group(1)

    else:

        cleaned_response = "GENERAL"

    print(

        f"\nLLM Route Decision: "
        f"{cleaned_response}\n"
    )

    return cleaned_response


# =========================================================
# ROUTE EXECUTION
# =========================================================

def execute_route(

    route,
    query
):

    print("\nExecuting Route Pipeline...\n")

    if route == "LEGAL_RAG":

        run_legal_pipeline(query)

    elif route == "UNSAFE":

        run_unsafe_handler(query)

    elif route == "GREETING":

        run_general_chat(query)

    else:

        run_general_chat(query)


# =========================================================
# TEST ROUTER
# =========================================================

def test_router():

    while True:

        query = input("\nEnter Query: ")

        route = classify_query(
            query
        )

        print("\nROUTER DECISION:\n")

        print({

            "route": route
        })

        execute_route(

            route,
            query
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    test_router()