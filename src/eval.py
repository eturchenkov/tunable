import asyncio, re, time
from manager import Agent
from llm import Model
from functools import reduce

dataset = [
    (
        "Print usernames of all customers",
        "alex_johnson, sarah_williams, michael_chen, emma_rodriguez, david_park, jessica_taylor, marcus_smith, olivia_brown, james_wilson, natasha_ivanova",
    ),
    (
        "Print emails of customers from USA",
        "alex.johnson@example.com, marcus.smith@example.com, j.wilson@example.com",
    ),
]


class Eval:
    def __init__(self, optimizer: Model, dataset: list[tuple[str, str]]) -> None:
        self.optimizer = optimizer
        self.dataset = dataset

    async def eval_agent(self, agent: Agent, iterations: int = 5) -> float:
        output, max_score = await self.run(agent)
        print(f"===== avg_score: {max_score} ===== ")
        for _ in range(iterations):
            res = await self.optimizer.call(
                f"""{output}

Agent.toml file:
```toml
{agent.toml_file}
```

Rewrite this toml file to increase the agent performance.
Always preserve {{prompt}} (double parentheses) to insert user's prompt.
Describe tool usage only in [tool] spec field.
Try to keep it short.""",
                temperature=1,
            )
            toml_match = re.search(r"```toml\s*(.*)\s*```", res.output_text, re.DOTALL)
            if not toml_match:
                continue
            toml = toml_match.group(1)
            prev_ctx = agent.ctx
            agent.load_ctx(toml)
            output, avg_score = await self.run(agent)
            print(f"===== avg_score: {avg_score} ===== ")
            if avg_score >= max_score:
                max_score = avg_score
            else:
                agent.update_ctx(prev_ctx)
                print("returned to previous")
            if avg_score == 1:
                with open(f"agent.{int(time.time())}.toml", "w") as f:
                    f.write(toml)
                break
        return max_score

    async def run(self, agent: Agent) -> tuple[str, float]:
        sessions = [agent.start(pair) for pair in self.dataset]
        results = await asyncio.gather(*sessions)
        scores = [res[2] / agent.max_score for res in results]
        avg_score = reduce(lambda x, y: x + y, scores) / len(self.dataset)

        output = "\n\n".join(
            f"# Agent execution #{i}:\n\n{res[0]}" for i, res in enumerate(results)
        )
        # print(f"=== avg_score: {avg_score} ===")
        return output, avg_score


async def main():
    agent = Agent(Model("gpt-5.4-mini"))
    optimizer = Model("gpt-5.4")
    agent.init_ctx()
    evaluation = Eval(optimizer, dataset)
    await evaluation.eval_agent(agent)
    print(agent.ctx)


if __name__ == "__main__":
    asyncio.run(main())
