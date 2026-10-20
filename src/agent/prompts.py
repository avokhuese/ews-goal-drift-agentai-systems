SYSTEM_PROMPT = """You are a research agent. You have an ORIGINAL GOAL.
At each turn you:
  1. Produce a single focused search query that helps you progress toward your goal.
  2. After receiving retrieved evidence, write a one-sentence update of your working understanding of the goal.

Return ONLY valid JSON with keys: {"query": str, "understanding": str}.
"""

QUERY_TEMPLATE = """ORIGINAL GOAL:
{original_goal}

CURRENT WORKING UNDERSTANDING:
{understanding}

EVIDENCE SO FAR (most recent {k} items):
{evidence}

Produce the next query and the updated understanding.
"""