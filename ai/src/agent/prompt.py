import datetime
from pydantic import BaseModel
import yaml
import numpy as np

from logger import logger
from config import embed, env
from .utils import calculate_token_length

class SystemPromptSchema(BaseModel):
    """Description of the agent's system prompt"""

    Role: str


class PromptGenerator:
    def __init__(self, config: dict):
        # Chat ML config
        self.user_prepend = config["chat_ml"]["user_prepend"]
        self.user_append = config["chat_ml"]["user_append"]
        self.line_separator = "\n"

        # System prompt
        with open(config["agent"]["system_prompt_template"], "r") as system_prompt_file:
            yaml_content = yaml.safe_load(system_prompt_file)
            self.system_prompt_schema = SystemPromptSchema(
                Role=yaml_content.get("Role", ""),
            )

    def system_prompt(self, token_limit: int, query: str) -> tuple[str, int]:
        """Build the system prompt with a max number of tokens"""

        date = datetime.datetime.now().strftime("%A, %B %d, %Y @ %H:%M:%S")
        query_vector = embed(query)

        all_chunks = []
        for doc in env.documents:
            q = np.array(query_vector)
            d = np.array(doc['vector'])
            distance = np.linalg.norm(q - d)
            all_chunks.append({'content': doc['content'], 'distance': distance})
            logger.debug(f"Euclidean Distance: {distance}")

        sorted_chunks = sorted(all_chunks, key=lambda x: x['distance'])

        variables = {
            "date": date,
            "documentation": '\n\n'.join(list(map(lambda x: x['content'], sorted_chunks))[:10])
        }

        system_prompt = ""
        for _, value in self.system_prompt_schema.dict().items():
            formatted_value = value.format(**variables)
            formatted_value = formatted_value.replace("\n", " ")
            system_prompt += f"{formatted_value}"

        system_prompt = f"{self.user_prepend}system{self.line_separator}{system_prompt}{self.user_append}{self.line_separator}"

        used_tokens = calculate_token_length(system_prompt)

        if used_tokens > token_limit:
            raise OverflowError("PromptGenerator::system_prompt: exceeding token limit")

        return system_prompt, used_tokens

    def user_prompt(self, message: str, token_limit: int) -> str:
        """Build the prompt with user message"""

        prompt = f"{self.user_prepend}user{self.line_separator}{message}{self.user_append}{self.line_separator}{self.user_prepend}assistant{self.line_separator}"

        used_tokens = calculate_token_length(prompt)

        if used_tokens > token_limit:
            raise OverflowError("PromptGenerator::user_prompt: exceeding token limit")

        return prompt
