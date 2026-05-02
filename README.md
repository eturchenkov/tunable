# Tunable

A Python framework for self-optimizing LLM agents. The agent runs against an evaluation dataset, and an optimizer LLM automatically rewrites the agent's configuration to improve its score — iterating until performance peaks or a target score is reached.

## How it works

The agent is driven by `agent.toml`, a plain-text config that defines its system prompt and tool usage instructions. At eval time:

1. The agent runs every dataset pair in parallel (two-stage LLM loop per pair).
2. Outputs are scored by cosine similarity against expected answers using OpenAI embeddings.
3. The optimizer LLM receives the full execution log and the current `agent.toml`, then rewrites it to improve the score.
4. The new config is kept only if the score improves; otherwise the previous config is restored.
5. When the agent hits a perfect score, the winning config is saved as `agent.<timestamp>.toml`.

## Running

```bash
# One-shot agent run
uv run python src/manager.py

# Evaluation + self-optimization loop
uv run python src/eval.py
```

Requires `OPENAI_API_KEY` in `.env`.

## agent.toml format

```toml
[instructions]
system = "... {{prompt}} ..."

[tool]
spec = "... tool usage instructions ..."
```

`{{prompt}}` is replaced with the user's query at runtime. The optimizer rewrites this file to improve agent performance.
