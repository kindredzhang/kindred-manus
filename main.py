import asyncio
from app.config import config
from app.logger import logger


async def main():
    # print(config.llm)
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    