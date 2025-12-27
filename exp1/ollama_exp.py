import logging
from pprint import pformat

from ollama import ChatResponse, chat

# create a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the chat with Gemma3 model...")

    response: ChatResponse = chat(
        model="gemma3:270m",
        messages=[
            {
                "role": "user",
                "content": "Why is the sky blue?",
            },
        ],
    )

    logger.info("Received response from the model.")
    logger.info(f"Response content:\n{pformat(response.model_dump())}")

    logger.info("Finished execution.")
