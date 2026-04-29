import os, asyncio, tomllib
from openai import AsyncOpenAI
from dotenv import load_dotenv
from user_db import parse

load_dotenv()


class Model:
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def call(self, ctx: str, temperature: float = 1.0):
        return await self.client.responses.create(
            model=self.model, input=ctx, temperature=temperature
        )


class Agent:
    def __init__(self, model):
        self.model = model
        self.toml_path = "agent.toml"
        self.ctx = ""
        self.max_score = 2

    def init_ctx(self):
        with open(self.toml_path, "rb") as f:
            data = tomllib.load(f)
        self.ctx = "\n\n".join([data["instructions"]["system"], data["tool"]["spec"]])

    def update_ctx(self, ctx: str):
        self.ctx = ctx

    async def start(self, prompt: str, log: bool = False) -> tuple[str, int]:
        score = 0
        ctx = self.ctx.replace("{{prompt}}", prompt)

        response = await self.model.call(ctx)
        output, s = parse(response.output_text)
        score += s

        ctx = f"{ctx}\n\n{response.output_text}\n\n{output}"

        response = await self.model.call(ctx)

        if log:
            print("=== ctx ===")
            print(f"{ctx}\n\n{response.output_text}")
            print("=== result ===")

        output, s = parse(response.output_text)
        score += s

        if log:
            print(output)
            print(f"=== score: {score} ===")
        return output, score


if __name__ == "__main__":
    agent = Agent(Model("gpt-5.4-mini"))
    agent.init_ctx()
    asyncio.run(agent.start("print usernames of all customers", True))
