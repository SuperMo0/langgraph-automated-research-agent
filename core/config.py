import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

EVALUATOR_PASS_THRESHOLD = 8.0
MAX_SYNTHESIS_ITERATIONS = 3
MAX_RESEARCH_ITERATIONS = 10
