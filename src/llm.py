import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class Model:
    def __init__(self, model: str):
        self.model = model
        self.client = client

    async def call(self, ctx: str, temperature: float = 1.0):
        return await self.client.responses.create(
            model=self.model, input=ctx, temperature=temperature
        )
