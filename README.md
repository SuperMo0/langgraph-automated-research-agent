# Automated Research Assistant

## how to run

```bash
# single research question
uv run research.py "What are the main differences between PydanticAI and LangGraph?"

# save the report to a custom directory
uv run research.py "How does vector search work?" --output reports/

# verbose mode, shows every agent tool call
uv run research.py "Who are the main players in the AI chip market?" --verbose
```

---

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-%23412991.svg?style=for-the-badge&logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-000000?style=for-the-badge)
![Typer](https://img.shields.io/badge/Typer-000000?style=for-the-badge&logo=fastapi&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=python&logoColor=white)

this project is part of my AI learning journey — it's the Chapter 2 capstone, pulling together every pattern from the chapter (router, parallel search, agentic loop, evaluator-optimizer, structured output) into one deployable system.

it's a CLI research assistant: give it a question and it classifies the query, runs an agentic research loop with parallel web search, drafts a structured report, and then scores its own report — retrying synthesis with feedback if the score isn't good enough.

I used LangGraph for the outer orchestration (router → research → synthesis → evaluator, with a conditional retry loop), but the research agent's tool-calling loop inside stage 2 is raw OpenAI API, no framework — it builds directly on the raw agent loop from an earlier project.

I used Typer for the CLI, and Pydantic for every structured output the LLM produces — the router's decision, the report itself, the evaluator's scores. nothing free-form gets trusted; it all goes through a typed schema, and the OpenAI tool schemas themselves are generated from those same Pydantic models rather than written by hand.

there's a single command: point it at a research question and it runs the full pipeline end to end.

flags

`--output` — directory to save the `.md` and `.json` report to, defaults to `reports/`  
`--verbose` — show every agent tool call during the research loop (tool name, truncated arguments, result length), off by default  

---

## setup

you need python and uv installed

```bash
git clone <repo>
cd langgraph101
uv sync
```

then create a `.env` file in the root with your api keys:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## how it works

the pipeline has four stages, wired together as a LangGraph state graph in `orchestration.py`:

1. **router** — classifies the query as `factual`, `analytical`, or `opinion`, which changes the system prompt the research agent gets.
2. **research agent** — a raw tool-calling loop (no framework) with five tools: `search_web`, `summarise_content`, `check_fact`, `get_current_date`, and `store_evidence`. when the model calls multiple tools in one turn (e.g. several searches at once), they run concurrently in a thread pool instead of one after another.
3. **synthesis** — reads everything stored via `store_evidence` and drafts a structured `ResearchReport` through a typed OpenAI structured-output call.
4. **evaluator** — scores the report on completeness, accuracy, clarity, and sourcing (1-10 each). if the average is below 8, it sends the report back to synthesis with feedback, up to 3 times.

every OpenAI and Tavily call goes through a shared, retry-wrapped client so transient rate limits or timeouts get retried automatically instead of crashing the run. if Tavily itself is down, the agent gets a `"search unavailable"` tool result back instead of the whole pipeline dying.

---

## project structure

```text
research.py           # cli entry point (typer)
orchestration.py       # langgraph state graph wiring the 4 stages together
router.py              # stage 1 — query classifier
research_agent.py      # stage 2 — raw openai tool-calling loop
synthesis.py           # stage 3 — evidence -> ResearchReport
evaluator.py           # stage 4 — scores a ResearchReport
models.py              # pydantic models shared across every stage
config.py              # env vars and pipeline constants
llm_client.py          # shared openai client + retry logic
search_client.py       # tavily wrapper with retries
evidence.py            # shared evidence-store formatting helper
report_output.py       # markdown/console rendering + file saving
reports/               # saved .md / .json reports (gitignored)
```
