# Tunable

Python framework for self-optimizing LLM agents with automated evaluation.

## Tech Stack

- **Python 3.13** with async/await throughout
- **OpenAI Responses API** — `model.call()` wraps `client.responses.create()`
- **uv** — package manager (`uv run`, `uv add`, `uv sync`)
- **numpy** — used in `emb.py` for cosine similarity

## Project Layout

```
src/
  llm.py           # Model class + shared AsyncOpenAI client
  manager.py       # Agent class — two-stage LLM loop, loads agent.toml
  user_db.py       # parse() — extract JSON from LLM output and score it
  users.py         # hardcoded dataset of 10 users (the "database")
  emb.py           # Embedding class — cosine similarity via OpenAI embeddings
  eval.py          # Eval class — runs agent + optimizer loop
  eval_dataset.py  # dataset of (prompt, expected_output) pairs
  instr.py         # stub (unused)
  agent.toml       # active agent config (instructions + tool spec)
  agent.*.toml     # saved snapshots from successful eval runs
```

## Running the Project

```bash
# One-shot agent run
uv run python src/manager.py

# Evaluation + optimization loop
uv run python src/eval.py
```

Both require `OPENAI_API_KEY` in `.env`.

## Key Classes

### `Model` (llm.py)
Thin wrapper around `AsyncOpenAI`. Reads `OPENAI_API_KEY` from env. Method: `call(ctx, temperature) -> Response`.

### `Agent` (manager.py)
Orchestrates a two-stage LLM loop driven by `agent.toml`:
1. `init_ctx()` — loads `[instructions].system` + `[tool].spec` from `agent.toml`; `{{prompt}}` is the user prompt placeholder
2. `start(pair)` → first LLM call → `parse()` (tool invocation + score) → second LLM call → returns `(execution_log, output, score)`
- Max score per run is 2 (one point per stage)
- `load_ctx(toml)` and `update_ctx(ctx)` allow the optimizer to hot-swap the agent config without restarting

### `Eval` (eval.py)
Two-level loop:
1. `run(agent)` — runs all dataset pairs in parallel via `asyncio.gather`, returns concatenated execution log + avg score
2. `eval_agent(agent, iterations)` — calls `run`, then asks the optimizer LLM to rewrite `agent.toml` to improve score; keeps the new config only if score improves; saves to `agent.<timestamp>.toml` when score hits 1.0

### `Embedding` (emb.py)
Wraps `text-embedding-3-small`. `calc()` fetches the vector; `distances(*rest)` returns cosine similarity scores in [0, 1].

### `parse()` (user_db.py)
Extracts a `````json``` block from LLM output. Returns `(result, score)`:
- `{"customers": ["field1", ...]}` → calls `shrink_list(users, fields)`, score 1
- `{"result": "..."}` → returns the string value, score 1
- Anything else → score 0

## agent.toml Format

```toml
[instructions]
system = "... {{prompt}} ..."

[tool]
spec = "... tool usage instructions ..."
```

`{{prompt}}` is replaced with the user's query at runtime.

## Known TODOs

- `eval_dataset.py` — expected outputs are all empty strings; fill them in for meaningful scoring
- `eval.py` — `Embedding` class exists but is not yet wired into the eval scoring pipeline
- `instr.py` — empty stub
- Model strings (`gpt-5.4-mini`, `gpt-5.4`) are placeholders; replace with valid model names before running
