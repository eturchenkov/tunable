import asyncio
import numpy as np
from llm import client


class Embedding:
    def __init__(self, text: str) -> None:
        self.text = text
        self.vec: np.ndarray

    async def calc(self) -> None:
        response = await client.embeddings.create(
            model="text-embedding-3-small", input=self.text, encoding_format="float"
        )
        self.vec = np.array(response.data[0].embedding)

    def distances(self, *rest: "Embedding") -> list[float]:
        return [self._cos_dist(self.vec, emb.vec) for emb in rest]

    def _cos_dist(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """it returns [0, 1] range"""
        cos_d = float(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )
        return round(cos_d, 2) if cos_d > 0 else 0


async def main():
    text = Embedding("it's my text")
    text2 = Embedding("12 + 10 - 500")
    text3 = Embedding("it's just a text")
    await asyncio.gather(text.calc(), text2.calc(), text3.calc())
    print(text.distances(text2, text3))


if __name__ == "__main__":
    asyncio.run(main())
