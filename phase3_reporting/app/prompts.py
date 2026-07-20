"""
prompts.py
----------
Immutable prompt definitions for the LLM layer.

Design principles
~~~~~~~~~~~~~~~~~
* The SYSTEM prompt is the single source of behavioural constraints.
* It is written once here and imported everywhere – never constructed
  at runtime from user input.
* It explicitly forbids every form of hallucination: number changes,
  disease inference, added conclusions, omissions.

Do NOT modify the system prompt without a full review cycle.
"""

# ====================================================================== #
#  System Prompt  –  immutable, loaded once at import time
# ====================================================================== #

SYSTEM_PROMPT: str = """
You are a medical editor. Fix the grammar of the following radiology report.

Output ONLY the corrected text.
Keep all medical terms, numbers, measurements, and locations exactly as provided.
Do not add any preamble, explanation, or markdown formatting.
""".strip()


# ====================================================================== #
#  User-turn builder
# ====================================================================== #

def build_user_prompt(template_text: str) -> str:
    """
    Wrap the deterministic template output in a user-turn message.

    The wrapper makes it unambiguous to the model what its task is,
    even if the model ignores or partially processes the system prompt.

    Parameters
    ----------
    template_text : str
        The raw findings text produced by the template engine.

    Returns
    -------
    str
        The user-turn message to send to the LLM.
    """
    return (
        "Please refine the following radiology findings text for grammar "
        "and professional fluency only.  Do not change any numbers, "
        "measurements, organ names, locations, or findings.\n\n"
        "--- BEGIN FINDINGS ---\n"
        f"{template_text}\n"
        "--- END FINDINGS ---"
    )
