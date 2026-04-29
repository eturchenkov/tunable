import asyncio, re
from manager import Agent, Model
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
    def __init__(self, agent: Agent, dataset: list[tuple[str, str]]) -> None:
        self.agent = agent
        self.dataset = dataset

    async def eval_session(self):
        pass

    async def run(self) -> tuple[str, float]:
        sessions = [self.agent.start(pair[0]) for pair in self.dataset]
        results = await asyncio.gather(*sessions)
        scores = [res[1] / self.agent.max_score for res in results]
        avg_score = reduce(lambda x, y: x + y, scores) / len(self.dataset)

        output = "\n\n".join(
            f"# Agent execution #{i}:\n\n{res[0]}" for i, res in enumerate(results)
        )
        print(f"=== avg_score: {avg_score} ===")
        return output, avg_score


async def main():
    agent = Agent(Model("gpt-5.4-mini"))
    agent.init_ctx()
    evaluation = Eval(agent, dataset)

    output, avg_score = await evaluation.run()
    print(avg_score)
    print(output)
    optimizer = Model("gpt-5.4")
    res = await optimizer.call(
        f"{output}\n\n{agent.ctx}\n\nRewrite toml", temperature=1.5
    )
    toml_match = re.search(r"```toml\s*(.*?)\s*```", res.output_text, re.DOTALL)
    if toml_match:
        ctx = toml_match.group(1)
        agent.update_ctx(ctx)
        print(ctx)


if __name__ == "__main__":
    asyncio.run(main())
