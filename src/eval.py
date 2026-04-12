import asyncio
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
        self.dataset = dataset  # todo:m add target_vec for each output

    async def eval_session(self):
        # todo:m get emb for output of the session
        pass

    async def run(self) -> float:
        sessions = [self.agent.start(pair[0]) for pair in self.dataset]
        results = await asyncio.gather(*sessions)
        scores = [res[1] / self.agent.max_score for res in results]
        avg_score = reduce(lambda x, y: x + y, scores) / len(self.dataset)
        print(f"=== avg_score: {avg_score} ===")
        return avg_score


async def main():
    agent = Agent(Model("gpt-5.4-mini"))
    await agent.init_ctx()
    evaluation = Eval(agent, dataset)
    await evaluation.run()


if __name__ == "__main__":
    asyncio.run(main())
