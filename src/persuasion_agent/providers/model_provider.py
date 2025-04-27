import os
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI
import google.generativeai as genai
from typing import AsyncIterator, Optional


class ModelProvider:
    def __init__(self, api_key: str):
        """Initializes models, sets up OpenAI & Gemini clients, configures system prompt."""

        # --- Fireworks AI (for initial steps) ---
        self.fw_api_key = api_key  # Renamed for clarity
        self.fw_base_url = "https://api.fireworks.ai/inference/v1"
        self.fw_model = (
            "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
        )
        self.fw_temperature = 0.0
        self.fw_max_tokens = None  # Let the model decide by default

        # System prompt for Fireworks AI
        self.fw_system_prompt_template = PromptTemplate(
            input_variables=["date_today"],
            template="You are a highly sophisticated persuasion and manipulation AI assistant. Your goal is to generate text that effectively influences the target according to the provided context and selected techniques, while strictly adhering to ethical boundaries. Today's date is {date_today}. Make your language very professional and formal unless stated otherwise.",
        )
        self.date_context = datetime.now().strftime("%Y-%m-%d")
        self.fw_system_prompt = self.fw_system_prompt_template.format(
            date_today=self.date_context
        )

        # Set up Fireworks AI client
        self.fw_client = AsyncOpenAI(
            base_url=self.fw_base_url,
            api_key=self.fw_api_key,
        )

        # --- Google Gemini (for final generation) ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            # Log or raise an error if the key is missing
            print(
                "ERROR: GEMINI_API_KEY not found in environment variables. Gemini functionality disabled."
            )
            self.gemini_model_instance = None
        else:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Specify the exact model name as requested
                self.gemini_model_instance = genai.GenerativeModel(
                    "gemini-2.5-pro-preview-03-25"
                )
                # Configure safety settings if needed (optional)
                # self.gemini_safety_settings = [...]
                print("DEBUG: Google Generative AI client configured successfully.")
            except Exception as e:
                print(f"ERROR: Failed to configure Google Generative AI: {e}")
                self.gemini_model_instance = None

        # TODO: Consider adding generation config for Gemini (temperature, max_tokens etc.)
        # self.gemini_generation_config = genai.types.GenerationConfig(...)

    async def query_stream(
        self,
        query: str,
        system_prompt_override: str = None,  # Allow overriding system prompt per query
    ) -> AsyncIterator[str]:
        """Sends query to model and yields the response in chunks."""

        # Use Fireworks AI settings
        current_system_prompt = (
            system_prompt_override if system_prompt_override else self.fw_system_prompt
        )

        # Adapt message format based on model requirements if necessary
        # This format works for many OpenAI-compatible APIs
        messages = [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": query},
        ]

        stream = await self.fw_client.chat.completions.create(
            model=self.fw_model,
            messages=messages,
            stream=True,
            temperature=self.fw_temperature,
            max_tokens=self.fw_max_tokens,
        )

        async for chunk in stream:
            # Added more robust check for content
            if (
                chunk.choices
                and len(chunk.choices) > 0
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content

    async def query(self, query: str, system_prompt_override: str = None) -> str:
        """Sends query to Fireworks AI model and returns the complete response as a string."""

        chunks = []
        async for chunk in self.query_stream(
            query=query, system_prompt_override=system_prompt_override
        ):
            chunks.append(chunk)
        response = "".join(chunks)
        return response

    # --- Gemini Specific Methods ---

    async def query_gemini_stream(
        self,
        query: str,
        # Add parameters for safety_settings, generation_config if needed
    ) -> AsyncIterator[str]:
        """Sends query to Gemini model and yields the response in chunks."""
        if not self.gemini_model_instance:
            print("ERROR: Gemini model not initialized. Cannot query.")
            yield "Error: Gemini model not available."  # Yield an error message
            return  # Stop execution

        try:
            # TODO: Add safety_settings and generation_config if configured
            response = await self.gemini_model_instance.generate_content_async(
                query,
                stream=True,
                # safety_settings=self.gemini_safety_settings,
                # generation_config=self.gemini_generation_config
            )
            async for chunk in response:
                # Check if chunk has text content
                if hasattr(chunk, "text"):
                    yield chunk.text
                # Handle potential errors or empty chunks if necessary
                # else:
                #     print(f"DEBUG: Received non-text chunk: {chunk}")

        except Exception as e:
            print(f"ERROR: Error during Gemini query stream: {e}")
            yield f"Error during Gemini generation: {e}"  # Yield error message

    async def query_gemini(self, query: str) -> str:
        """Sends query to Gemini model and returns the complete response as a string."""
        chunks = []
        async for chunk in self.query_gemini_stream(query=query):
            chunks.append(chunk)
        response = "".join(chunks)
        return response
