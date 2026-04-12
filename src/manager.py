import os, asyncio, aiofiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from user_db import parse

load_dotenv()


class Model:
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def call(self, ctx: str):
        return await self.client.responses.create(model=self.model, input=ctx)


class Agent:
    def __init__(self, model):
        self.model = model
        self.file_paths = ["instr.tl", "user_db.tl"]
        self.files = []
        self.max_score = 2

    async def read_file(self, filename: str):
        async with aiofiles.open(filename, mode="r") as f:
            contents = await f.read()
        return contents

    async def init_ctx(self):
        tasks = [self.read_file(path) for path in self.file_paths]
        self.files = await asyncio.gather(*tasks)

    async def start(self, prompt: str) -> tuple[str, int]:
        score = 0
        ctx = "\n\n".join(self.files).replace("{{prompt}}", prompt)

        response = await self.model.call(ctx)
        output, s = parse(response.output_text)
        score += s

        ctx = f"{ctx}\n\n{response.output_text}\n\n{output}"

        response = await self.model.call(ctx)

        print("=== ctx ===")  # send it with templates to large llm to rewrite templates
        print(f"{ctx}\n\n{response.output_text}")
        print("=== result ===")  # use it to calc score

        output, s = parse(response.output_text)
        score += s

        print(output)
        print(f"=== score: {score} ===")
        return output, score


if __name__ == "__main__":
    agent = Agent(Model("gpt-5.4-mini"))
    asyncio.run(agent.init_ctx())
    asyncio.run(agent.start("print usernames of all customers"))
