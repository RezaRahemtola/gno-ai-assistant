import aiohttp

from logger import logger
from config import env
from .prompt import PromptGenerator
from .utils import calculate_token_length


class Agent:
    def __init__(self, config: dict):
        # Model
        self.model_api_url: str = config["model"]["api_url"]
        self.max_prompt_tokens: int = config["model"]["max_prompt_tokens"]
        self.max_completion_tokens: int = config["model"]["max_completion_tokens"]
        self.temperature: float = config["model"]["temperature"]
        self.top_p: float = config["model"]["top_p"]
        self.top_k: float = config["model"]["top_k"]

        # Agent
        self.max_completion_tries = config["agent"]["max_completion_tries"]
        self.stop_sequences = config["chat_ml"]["stop_sequences"]

        # Utils
        self.prompt_generator = PromptGenerator(config)

    async def generate_prompt(self, message: str) -> str:
        """Generate the prompt within the model's context window"""

        system_prompt, used_system_tokens = self.prompt_generator.system_prompt(self.max_prompt_tokens)

        user_prompt = self.prompt_generator.user_prompt(
            message, token_limit=self.max_prompt_tokens - used_system_tokens
        )

        return f"{system_prompt}{user_prompt}"

    async def complete(self, prompt: str) -> tuple[str, int]:
        """Complete on a prompt with the model"""

        session = aiohttp.ClientSession()

        params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,

            # llamacpp params
            "n_predict": self.max_completion_tokens,
            "typical_p": 1,
            "tfs_z": 1,
            "stop": self.stop_sequences,
            "cache_prompt": True,
            "use_default_badwordsids": False,
        }

        tries = 0
        errors = []
        full_result = ""
        result = ""
        while tries < self.max_completion_tries:
            # Append the compound result to the prompt
            params["prompt"] = f"{params['prompt']}{result}"
            try:
                async with session.post(self.model_api_url, json=params) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        result = response_data["content"]
                        full_result = f"{full_result}{result}"
                        token_count = calculate_token_length(full_result)
                        await session.close()
                        return full_result, token_count
                    else:
                        raise RuntimeError(f"Agent::complete: Request failed: {response.status}")
            except Exception as e:
                logger.debug(f"Agent::complete: Error completing prompt: {e}")
                errors.append(e)
            finally:
                tries += 1
        raise RuntimeError(f"Agent::complete: Failed to complete prompt: {errors}")


    async def yield_response(self, message: str):
        """Yield a string containing the agent response"""

        # Build the prompt
        prompt = await self.generate_prompt(message)

        logger.debug(f"Agent::yield_response: prompt: {prompt}")

        completion, _tokens = await self.complete(prompt)

        logger.debug(f"Agent::yield_response: completion: {completion}")

        yield completion
