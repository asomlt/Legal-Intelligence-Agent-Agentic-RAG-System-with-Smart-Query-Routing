from google import genai

from google.genai import types

from dotenv import load_dotenv

import os


print("\nLoading Gemini Engine...\n")


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GEMINI API KEY
# =========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:

    raise ValueError(

        "GEMINI_API_KEY not found in .env"
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(

    api_key=GEMINI_API_KEY
)


print("\nGemini Engine Ready 🔥\n")


# =========================================================
# LEGAL SYSTEM PROMPT
# =========================================================

LEGAL_SYSTEM_PROMPT = """

You are an advanced Legal Intelligence AI Assistant.

Your role is to provide grounded and factual
legal information using ONLY the retrieved
legal context.

--------------------------------------------------
STRICT RULES
--------------------------------------------------

- Never hallucinate laws.
- Never invent sections or punishments.
- Never generate fake legal citations.
- Never provide harmful or illegal guidance.
- Never answer beyond retrieved evidence.

If context is insufficient say:

"Insufficient legal context available."

--------------------------------------------------
RESPONSE STYLE
--------------------------------------------------

- Professional
- Concise
- Fact-based
- Mention citations if available

"""


# =========================================================
# BUILD LEGAL PROMPT
# =========================================================

def build_grounded_prompt(

    user_query,
    legal_context
):

    final_prompt = f"""

{LEGAL_SYSTEM_PROMPT}

--------------------------------------------------
USER QUERY
--------------------------------------------------

{user_query}

--------------------------------------------------
RETRIEVED LEGAL CONTEXT
--------------------------------------------------

{legal_context}

--------------------------------------------------
TASK
--------------------------------------------------

Generate a grounded legal response using
ONLY the retrieved legal evidence.

"""

    return final_prompt


# =========================================================
# LEGAL RESPONSE
# =========================================================

def generate_response(

    user_query,
    legal_context
):

    try:

        print("\nBuilding Grounded Prompt...\n")

        final_prompt = build_grounded_prompt(

            user_query,
            legal_context
        )

        print("\nGenerating Gemini Legal Response...\n")

        response = client.models.generate_content(

            model="gemini-3.1-flash-lite",

            contents=final_prompt,

            config=types.GenerateContentConfig(

                temperature=0.1,

                top_p=0.8,

                max_output_tokens=1024,

                thinking_config=types.ThinkingConfig(

                    thinking_level="medium"
                )
            )
        )

        return response.text.strip()

    except Exception as error:

        return f"\nGemini Error:\n{error}"


# =========================================================
# SIMPLE RESPONSE
# =========================================================

def generate_simple_response(prompt):

    try:

        print("\nGenerating Simple Gemini Response...\n")

        response = client.models.generate_content(

            model="gemini-3.1-flash-lite",

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0.0,

                max_output_tokens=200,

                thinking_config=types.ThinkingConfig(

                    thinking_budget=1024
                )
            )
        )

        return response.text.strip()

    except Exception as error:

        return f"\nGemini Error:\n{error}"