import os, asyncio, aiofiles
from openai import AsyncOpenAI
from dotenv import load_dotenv
from user_db import parse

load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def call(ctx: str):
    return await client.responses.create(model="gpt-5.4-mini", input=ctx)


async def read_file(filename: str):
    async with aiofiles.open(filename, mode="r") as f:
        contents = await f.read()
    return contents


async def run_agent(prompt: str):
    file_paths = ["instr.tl", "user_db.tl"]
    tasks = [read_file(path) for path in file_paths]
    files = await asyncio.gather(*tasks)
    ctx = "\n\n".join(files).replace("{{prompt}}", prompt)
    response = await call(ctx)
    ctx = f"{ctx}\n\n{response.output_text}\n\n{parse(response.output_text)}"
    response = await call(ctx)
    print("=== ctx ===")  # send it with templates to large llm to rewrite templates
    print(f"{ctx}\n\n{response.output_text}")
    print("=== result ===")  # use it to calc score
    print(parse(response.output_text))


if __name__ == "__main__":
    asyncio.run(run_agent("print usernames of all customers"))
