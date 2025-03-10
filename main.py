import asyncio
from app.config import config


async def main():
    print(config.llm)

if __name__ == "__main__":
    asyncio.run(main())
    