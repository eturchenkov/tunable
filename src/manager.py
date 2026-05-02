import asyncio, tomllib
from llm import Model
from user_db import parse


class Agent:
    def __init__(self, model):
        self.model = model
        self.toml_path = "agent.toml"
        self.toml_file = ""
        self.ctx = ""
        self.max_score = 2

    def init_ctx(self):
        with open(self.toml_path, "rb") as f:
            self.toml_file = f.read().decode("utf-8")
            data = tomllib.loads(self.toml_file)
        self.ctx = "\n\n".join([data["instructions"]["system"], data["tool"]["spec"]])

    def update_ctx(self, ctx: str):
        self.ctx = ctx

    def load_ctx(self, toml: str):
        data = tomllib.loads(toml)
        self.ctx = "\n\n".join([data["instructions"]["system"], data["tool"]["spec"]])

    async def start(
        self, pair: tuple[str, str], log: bool = False
    ) -> tuple[str, str, int]:
        score = 0
        execution = ""
        prompt, target = pair
        ctx = self.ctx.replace("{{prompt}}", prompt)

        response = await self.model.call(ctx)
        output, s = parse(response.output_text)
        execution = f"{ctx}\n\nResponse #1:\n\n{response.output_text}"
        score += s

        ctx = f"{ctx}\n\n{response.output_text}\n\n{output}"

        response = await self.model.call(ctx)
        execution += f"\n\nResponse #2:\n\n{response.output_text}"

        if log:
            print("=== ctx ===")
            print(f"{ctx}\n\n{response.output_text}")
            print("=== result ===")

        output, s = parse(response.output_text)
        score += s
        execution += f"\n\nOutput:\n\n{output}\n\nTarget:\n\n{target}"

        if log:
            print(output)
            print(f"=== score: {score} ===")
        return execution, output, score


if __name__ == "__main__":
    agent = Agent(Model("gpt-5.4-mini"))
    agent.init_ctx()
    asyncio.run(agent.start(("print usernames of all customers", ""), True))
