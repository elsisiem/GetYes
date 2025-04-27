# /persuasion_agent/providers/model_provider.py

import google.generativeai as genai
import logging
import os
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any, Union
import asyncio
import json
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.types import (
    BlockedPromptException,
    StopCandidateException,
)  # Corrected exception names

logger = logging.getLogger(__name__)

# Generation Config Defaults
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 64,
    "response_mime_type": "text/plain",
}

# Safety Settings - Using ENUMS
DEFAULT_SAFETY_SETTINGS = {}


class GeminiModelProvider:
    """Provides interface to Google Gemini models, supporting different models."""

    def __init__(self, api_key: str):
        """Initializes the Gemini client and configures model names."""
        if not api_key:
            raise ValueError("Gemini API key is required.")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

        # Define Model Names (Use env vars or fall back to defaults)
        self.flash_model_name = os.getenv(
            "GEMINI_FLASH_MODEL", "gemini-2.5-flash-preview-04-17"
        )
        self.pro_model_name = os.getenv(
            "GEMINI_PRO_MODEL", "gemini-2.5-pro-preview-03-25"
        )

        # --- UPDATED PERSONA ---
        self.base_system_instruction = f"""You are GetYes, a highly sophisticated persuasion AI assistant. Your goal is to help users craft compelling messages to achieve their objectives. You analyze context, suggest techniques, and generate persuasive text, always adhering to the ethical boundaries defined by the user. Today's date is {datetime.now().strftime("%Y-%m-%d")}. Maintain a helpful and professional tone unless the user requests otherwise."""
        # --- END UPDATE ---

        # Cache for model instances
        self._models: Dict[str, genai.GenerativeModel] = {}

        logger.info(
            f"GeminiModelProvider initialized. Flash model: '{self.flash_model_name}', Pro model: '{self.pro_model_name}'"
        )

    def _get_model(
        self, model_name: str, system_instruction_override: Optional[str] = None
    ) -> genai.GenerativeModel:
        """Gets or creates the GenerativeModel instance for the specified model name."""
        # --- UPDATED SYSTEM PROMPT REFERENCE ---
        # Use the agent's name "GetYes" in the system prompt unless overridden
        default_system_instruction = self.base_system_instruction  # Now uses GetYes
        current_system_instruction = (
            system_instruction_override or default_system_instruction
        )
        # --- END UPDATE ---

        cache_key = f"{model_name}_{hash(current_system_instruction)}"

        if cache_key not in self._models:
            logger.info(
                f"Creating new GenerativeModel instance for '{model_name}' with specified system instruction."
            )
            try:
                self._models[cache_key] = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=current_system_instruction,
                    safety_settings=DEFAULT_SAFETY_SETTINGS,
                )
            except Exception as e:
                logger.critical(
                    f"Failed to create GenerativeModel instance '{model_name}': {e}",
                    exc_info=True,
                )
                if "not found" in str(e).lower() or "is not found" in str(e).lower():
                    logger.error(
                        f"Model name '{model_name}' might be incorrect or not available for your API key/region."
                    )
                raise
        return self._models[cache_key]

    async def _generate_with_retry(
        self,
        model_name: str,
        prompt: str,
        system_prompt_override: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        max_retries: int = 2,
        initial_delay: float = 1.0,
    ) -> Union[AsyncIterator[str], str]:
        """Internal method to handle content generation with retries for specific errors."""
        model = self._get_model(model_name, system_prompt_override)
        effective_config_dict = {
            **DEFAULT_GENERATION_CONFIG,
            **(generation_config or {}),
        }
        effective_config = genai.types.GenerationConfig(**effective_config_dict)
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt+1} calling model '{model_name}'...")
                if stream:
                    response_stream = await model.generate_content_async(
                        prompt,
                        generation_config=effective_config,
                        stream=True,
                    )

                    async def _stream_wrapper(response_stream):
                        try:
                            async for chunk in response_stream:
                                if chunk.parts:
                                    yield chunk.text
                                elif (
                                    hasattr(chunk, "prompt_feedback")
                                    and chunk.prompt_feedback
                                    and chunk.prompt_feedback.block_reason
                                ):
                                    reason = chunk.prompt_feedback.block_reason.name
                                    logger.warning(f"Stream blocked: {reason}")
                                    yield f"[ERROR: Blocked by filter ({reason})]"
                                    break
                                elif (
                                    hasattr(chunk, "candidates")
                                    and chunk.candidates
                                    and chunk.candidates[0].finish_reason.name != "STOP"
                                ):
                                    reason = chunk.candidates[0].finish_reason.name
                                    logger.warning(f"Stream stopped: {reason}")
                                    yield f"[ERROR: Stopped ({reason})]"
                                    break
                                else:
                                    logger.debug(f"Empty chunk: {chunk}")
                        except (BlockedPromptException, StopCandidateException) as e:
                            logger.warning(f"Blocked (stream): {e}")
                            yield f"[ERROR: Blocked: {e}]"
                        except Exception as e:
                            logger.error(f"Stream error: {e}", exc_info=True)
                            yield f"[ERROR: Streaming error: {e}]"

                    return _stream_wrapper(response_stream)
                else:  # Non-streaming
                    response = await model.generate_content_async(
                        prompt, generation_config=effective_config, stream=False
                    )
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        reason = response.prompt_feedback.block_reason.name
                        logger.warning(f"Prompt blocked: {reason}")
                        raise BlockedPromptException(f"Prompt blocked: {reason}")
                    if not response.candidates:
                        reason = "Unknown (No Candidates)"
                        if (
                            hasattr(response, "prompt_feedback")
                            and response.prompt_feedback
                            and hasattr(response.prompt_feedback, "finish_reason")
                        ):
                            reason = response.prompt_feedback.finish_reason.name
                        logger.warning(
                            f"No candidates. Reason: {reason}. Resp: {response}"
                        )
                        if "SAFETY" in reason.upper():
                            raise BlockedPromptException("Blocked (inferred).")
                        else:
                            return f"[ERROR: No candidates. Reason: {reason}]"
                    candidate = response.candidates[0]
                    reason = candidate.finish_reason.name
                    if reason != "STOP":
                        logger.warning(f"Finished reason: {reason}. Resp: {response}")
                        if reason == "SAFETY":
                            raise BlockedPromptException("Blocked.")
                        elif reason == "RECITATION":
                            raise StopCandidateException("Stopped (recitation).")
                        elif reason == "MAX_TOKENS":
                            logger.warning("Truncated (max_tokens).")
                        else:
                            return f"[ERROR: Unexpected finish reason: {reason}]"
                    if not candidate.content or not candidate.content.parts:
                        logger.warning(
                            f"No content parts. Reason: {reason}. Resp: {response}"
                        )
                        return f"[ERROR: No content. Reason: {reason}]"
                    return response.text
            except (BlockedPromptException, StopCandidateException) as e:
                logger.warning(f"Blocked '{model_name}' (att {attempt + 1}): {e}")
                raise e
            except genai.errors.GoogleAPIError as e:
                logger.warning(f"API Error '{model_name}' (att {attempt + 1}): {e}")
                last_exception = e
                if isinstance(e, genai.errors.NotFound):
                    logger.error(f"Model '{model_name}' not found.")
                    raise e
                if attempt >= max_retries:
                    logger.error(
                        f"API call failed after {max_retries + 1} attempts for '{model_name}'."
                    )
                    raise last_exception
                else:
                    delay = initial_delay * (2**attempt)
                    logger.info(f"Retrying '{model_name}' in {delay:.2f}s...")
                    await asyncio.sleep(delay)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Unexpected failure '{model_name}' (att {attempt + 1}): {e}. Retrying in {initial_delay * (2**attempt):.2f}s..."
                )
                if attempt >= max_retries:
                    logger.error(
                        f"Call failed after {max_retries + 1} attempts for '{model_name}'."
                    )
                    raise last_exception
                else:
                    await asyncio.sleep(initial_delay * (2**attempt))

    async def query_stream(
        self,
        model_name: str,
        query: str,
        system_prompt_override: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Sends query to the specified model and yields response chunks."""
        logger.debug(f"Streaming query to model: '{model_name}'.")
        try:
            stream = await self._generate_with_retry(
                model_name=model_name,
                prompt=query,
                system_prompt_override=system_prompt_override,
                generation_config=generation_config,
                stream=True,
            )
            async for chunk in stream:
                yield chunk
        except (BlockedPromptException, StopCandidateException) as e:
            logger.error(f"Stream failed '{model_name}' (blocking): {e}")
            yield f"[ERROR: Blocked: {e}]"
        except Exception as e:
            logger.error(f"Stream failed '{model_name}': {e}", exc_info=True)
            yield f"[ERROR: Streaming error: {e}]"

    async def query(
        self,
        model_name: str,
        query: str,
        system_prompt_override: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        response_format: str = "text",
    ) -> str:
        """Sends query to the specified model and returns the complete response."""
        logger.debug(f"Querying model: '{model_name}' (format: {response_format}).")
        query_generation_config = generation_config or {}
        if response_format == "json":
            query_generation_config["response_mime_type"] = "application/json"
        elif "response_mime_type" in query_generation_config:
            query_generation_config["response_mime_type"] = "text/plain"

        try:
            response_text = await self._generate_with_retry(
                model_name=model_name,
                prompt=query,
                system_prompt_override=system_prompt_override,
                generation_config=query_generation_config,
                stream=False,
            )
            if response_format == "json":
                parsed = self.extract_json_from_response(response_text)
                if parsed is None:
                    logger.error(
                        f"Invalid JSON from '{model_name}': {response_text[:500]}"
                    )
                    return json.dumps(
                        {
                            "error": "Invalid JSON from AI",
                            "received_text": response_text,
                        }
                    )
                else:
                    return response_text
            else:
                return response_text
        except (BlockedPromptException, StopCandidateException) as e:
            logger.error(f"Query failed '{model_name}' (blocking): {e}")
            error_payload = {"error": "Blocked", "details": str(e)}
            if response_format == "json":
                try:
                    return json.dumps(error_payload)
                except TypeError:
                    return json.dumps(
                        {"error": "Blocked", "details": "Serialization error"}
                    )
            else:
                return f"[ERROR: Blocked: {e}]"
        except Exception as e:
            logger.error(f"Query failed '{model_name}': {e}", exc_info=True)
            error_payload = {"error": "Unexpected query error.", "details": str(e)}
            if response_format == "json":
                try:
                    return json.dumps(error_payload)
                except TypeError:
                    return json.dumps(
                        {
                            "error": "Unexpected query error.",
                            "details": "Serialization error",
                        }
                    )
            else:
                return f"[ERROR: Unexpected query error: {e}]"

    @staticmethod
    def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
        """Attempts to extract JSON object from a string."""
        logger.debug(f"Extracting JSON from: {response_text[:200]}...")
        json_match = re.search(
            r"```json\s*(\{[\s\S]*?\})\s*```", response_text, re.IGNORECASE
        )
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug("Found JSON in markdown.")
        else:
            clean_response = response_text.strip()
            if clean_response.startswith("{") and clean_response.endswith("}"):
                json_str = clean_response
                logger.debug("Assuming full response is JSON.")
            else:
                logger.debug("No clear JSON structure found.")
                return None
        try:
            parsed_json = json.loads(json_str)
            logger.debug("Parsed JSON successfully.")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed: {e}. String: {json_str}")
            return None
