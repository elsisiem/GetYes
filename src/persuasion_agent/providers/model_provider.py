from datetime import datetime
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI
from typing import AsyncIterator

class ModelProvider:
    def __init__(
        self,
        api_key: str
    ):
        """ Initializes model, sets up OpenAI client, configures system prompt."""

        # Model provider API key
        self.api_key = api_key
        # Model provider URL - TODO: Consider making this configurable via .env
        self.base_url = "https://api.fireworks.ai/inference/v1"
        # Identifier for specific model that should be used - TODO: Make configurable
        self.model = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
        # Temperature setting for response randomness - TODO: Make configurable
        self.temperature = 0.0
        # Maximum number of tokens for responses - TODO: Make configurable
        self.max_tokens = None # Let the model decide by default
        # System prompt - Can be overridden or customized later
        self.system_prompt_template = PromptTemplate(
            input_variables=["date_today"],
            template="You are a highly sophisticated persuasion and manipulation AI assistant. Your goal is to generate text that effectively influences the target according to the provided context and selected techniques, while strictly adhering to ethical boundaries. Today's date is {date_today}."
        )
        self.date_context = datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = self.system_prompt_template.format(date_today=self.date_context)


        # Set up model API client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )


    async def query_stream(
        self,
        query: str,
        system_prompt_override: str = None # Allow overriding system prompt per query
    ) -> AsyncIterator[str]:
        """Sends query to model and yields the response in chunks."""

        current_system_prompt = system_prompt_override if system_prompt_override else self.system_prompt

        # Adapt message format based on model requirements if necessary
        # This format works for many OpenAI-compatible APIs
        messages = [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": query}
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


    async def query(
        self,
        query: str,
        system_prompt_override: str = None
    ) -> str:
        """Sends query to model and returns the complete response as a string."""

        chunks = []
        async for chunk in self.query_stream(query=query, system_prompt_override=system_prompt_override):
            chunks.append(chunk)
        response = "".join(chunks)
        return response