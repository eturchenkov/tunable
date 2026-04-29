# Tunable

Python framework to make tunable agents with evaluation score.

## Tech Stack

- **Python 3.13** with async/await throughout
- **OpenAI Responses API** — `model.call()` wraps `client.responses.create()`
- **uv** — package manager (`uv run`, `uv add`, `uv sync`)
- **pinecone-client** + **numpy** — imported, not yet wired up (planned for embedding-based eval)

## Project Layout

```
src/
  manager.py       # Agent and Model classes — core orchestration
  user_db.py       # parse() — extract JSON from LLM output and score it
  users.py         # hardcoded dataset of 10 users (the "database")
  eval.py          # Eval class — runs agent against test cases in parallel
  eval_dataset.py  # list of (prompt, expected_output) pairs
  instr.py         # stub (unused)
```

## Running the Project

```bash
# One-shot agent run
uv run python src/manager.py

# Evaluation suite
uv run python src/eval.py
```

Both require `OPENAI_API_KEY` in `.env`.

## Key Classes

### `Model` (manager.py)
Thin wrapper around `AsyncOpenAI`. Reads `OPENAI_API_KEY` from env. Single method: `call(ctx: str) -> str`.

### `Agent` (manager.py)
Orchestrates a two-stage LLM loop:
1. Load `instr.tl` + `user_db.tl` templates via `init_ctx()`
2. `start(prompt)` → first LLM call → `parse()` to score → second LLM call → return `(output, score)`

### `Eval` (eval.py)
Runs all `(prompt, expected_output)` pairs from `eval_dataset.py` in parallel via `asyncio.gather`, averages scores. TODOs in the file indicate planned embedding-based scoring to replace the current regex scoring.

### `parse()` (user_db.py)
Extracts JSON from freeform LLM text using regex. Returns `(text, score)` where score is 1 if the JSON matches the expected schema (`{"customers": [...]}` or `{"result": "..."}`), 0 otherwise.

## Environment Variables

| Variable | Purpose |
| `OPENAI_API_KEY` | Required — OpenAI API access |

## Model Name

The current model string in `manager.py` (`gpt-5.4-mini`) is a placeholder and will fail at runtime. Update it to a valid model name (e.g. `gpt-4o-mini`) before running.

## Current State & Known TODOs

- `eval_dataset.py` — all expected outputs are empty strings; fill these in to make eval meaningful
- `eval.py` — two `todo:m` comments flag planned embedding-based scoring (Pinecone + numpy)
- `instr.py` — empty stub, likely intended for instruction management
- `manager.py` line 46 — comment indicates a planned "large LLM rewrite templates" stage
- `.env` should not be committed; it's currently tracked despite being in `.gitignore`
