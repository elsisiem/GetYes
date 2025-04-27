import logging
import os
import json
from dotenv import load_dotenv
from typing import AsyncIterator, List, Dict, Any
from .sentiment import sentiment_analysis

# Assuming ModelProvider exists or will be created similar to search_agent
# If it's generic enough, we might move it to a shared location later.
# For now, let's assume it's specific to this agent's providers.
# We'll need to create this file/directory structure.
from .providers.model_provider import ModelProvider  # Reverted to relative import

from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)

# Load environment variables from .env file in the parent 'persuasion_agent' directory
dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")  # Adjusted path
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)
# Configure logging level (e.g., INFO, DEBUG)
logging.basicConfig(level=logging.INFO)


class PersuasionAgent(AbstractAgent):
    def __init__(self, name: str = "Persuasion Agent"):
        super().__init__(name)

        # --- Configuration & Initialization ---
        # 1. Load Model API Key
        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            logger.error("MODEL_API_KEY environment variable not found.")
            raise ValueError("MODEL_API_KEY is not set in the environment")
        # Assuming ModelProvider needs the API key
        self._model_provider = ModelProvider(api_key=model_api_key)
        logger.info("ModelProvider initialized.")

        # 2. Load Techniques Database
        # Load from the parent 'persuasion_agent' directory
        db_path = os.path.join(
            os.path.dirname(__file__), "../../techniques_database.json"  # Adjusted path
        )
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self._techniques_db: List[Dict[str, Any]] = json.load(f)
            logger.info(f"Techniques database loaded successfully from {db_path}.")
            if not isinstance(self._techniques_db, list):
                logger.warning("Techniques database is not a list.")
                self._techniques_db = []  # Ensure it's a list
        except FileNotFoundError:
            logger.error(f"Techniques database file not found at {db_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from techniques database at {db_path}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading techniques database: {e}"
            )
            raise

    async def _get_technique_summaries(self) -> str:
        """Generates a string containing summaries of all techniques."""
        summaries = []
        for technique in self._techniques_db:
            name = technique.get("name", "Unnamed Technique")
            # Create a brief principle summary - using keywords for now
            keywords = ", ".join(technique.get("keywords", []))
            summary = f"Name: {name}\nKeywords: {keywords}\n---"
            summaries.append(summary)
        return "\n".join(summaries)

    async def assist(
        self, session: Session, query: Query, response_handler: ResponseHandler
    ):
        """
        Main method to handle user requests for persuasion.
        Implements Phase 2 of the plan.
        """
        # Use session.activity_id for logging
        logger.info(
            f"Received query for activity {session.activity_id}: {query.prompt[:100]}..."
        )
        user_context = query.prompt

        try:
            # --- Step 1: Advanced Technique Selection (LLM Call 1) ---
            await response_handler.emit_text_block(
                "STATUS", "Analyzing context and selecting techniques..."
            )
            # Extract exact names for the prompt constraint
            exact_technique_names = [
                t.get("name", "") for t in self._techniques_db if t.get("name")
            ]
            exact_names_list_str = "\n".join(
                [f'- "{name}"' for name in exact_technique_names]
            )

            # No longer need summaries, just the exact names list
            # logger.info("Generating technique summaries...")
            # technique_summaries = await self._get_technique_summaries() # Removed summary generation

            selection_prompt = f"""
Analyze the following user context and select the top 1-3 most effective persuasion/manipulation techniques from the provided list to achieve the user's goal. Consider the target profile, relationship, context, and desired outcome.

**User Context:**
{user_context}

**Available Technique Names:**
{exact_names_list_str}

**Instruction:** Choose up to 3 techniques from the **Available Technique Names** list above that are most suitable for the User Context. Respond ONLY with the exact names of the selected techniques as they appear in the list, separated by commas (e.g., "Technique Name 1, Technique Name 2"). Do not include any other text, explanation, or formatting. Ensure the names match exactly.
"""
            logger.debug(f"Technique selection prompt:\n{selection_prompt}")

            # Use non-streaming query for this call
            selected_names_str = await self._model_provider.query(selection_prompt)
            logger.info(f"LLM response for technique selection: '{selected_names_str}'")

            # Parse the response
            selected_technique_names = [
                name.strip() for name in selected_names_str.split(",") if name.strip()
            ]

            if not selected_technique_names:
                logger.warning("LLM did not return valid technique names.")
                # Handle error - maybe default techniques or inform user?
                await response_handler.emit_text_block(
                    "ERROR", "Could not select appropriate techniques."
                )
                await response_handler.complete()
                return

            logger.info(f"Selected techniques: {selected_technique_names}")
            await response_handler.emit_json(
                "SELECTED_TECHNIQUES", {"techniques": selected_technique_names}
            )

            # --- Step 2: Application Analysis Generation (LLM Call 2) ---
            await response_handler.emit_text_block(
                "STATUS", "Generating application analysis..."
            )
            logger.info(
                "Attempting to retrieve details for selected techniques..."
            )  # Added log

            selected_technique_details = []
            for name in selected_technique_names:
                found = False
                for technique in self._techniques_db:
                    if technique.get("name") == name:
                        selected_technique_details.append(
                            {
                                "name": name,
                                "instructions": technique.get(
                                    "instructions", "No instructions provided."
                                ),
                                "example": technique.get(
                                    "example", "No example provided."
                                ),
                            }
                        )
                        found = True
                        break
                if not found:
                    logger.warning(f"Details not found for selected technique: {name}")
                    # Optionally skip or handle missing details

            if not selected_technique_details:
                logger.error("Could not retrieve details for any selected technique.")
                await response_handler.emit_text_block(
                    "ERROR", "Failed to retrieve technique details."
                )
                await response_handler.complete()
                return

            # Format details for the prompt
            formatted_details = "\n\n".join(
                [
                    f"**Technique: {details['name']}**\nInstructions:\n{details['instructions']}\nExample:\n{details['example']}"
                    for details in selected_technique_details
                ]
            )
            logger.info(
                "Successfully retrieved and formatted technique details."
            )  # Added log

            analysis_prompt = f"""
Analyze the provided user context and the selected persuasion techniques. Generate a concise analysis explaining HOW to specifically apply EACH selected technique to the given user context for maximum persuasive/manipulative effect. Focus on actionable advice tailored to the situation. Adhere strictly to any ethical boundaries mentioned in the user context.

**User Context:**
{user_context}

**Selected Techniques & Details:**
{formatted_details}

**Instruction:** Generate the application analysis. Be specific and practical.
"""
            logger.debug(
                f"Application analysis prompt (approx length {len(analysis_prompt)}):\n{analysis_prompt}"
            )  # Added length log

            # Use non-streaming query for the analysis
            logger.info("Making LLM call 2 for application analysis...")  # Added log
            application_analysis = await self._model_provider.query(analysis_prompt)
            logger.info(
                "LLM call 2 completed. Generated application analysis."
            )  # Added log
            logger.debug(f"Application Analysis:\n{application_analysis}")

            # --- Step 3: Final Prompt Generation & Streaming (LLM Call 3) ---
            await response_handler.emit_text_block(
                "STATUS", "Generating persuasive text..."
            )
            logger.info("Constructing final prompt for persuasive text generation...")

            final_prompt = f"""
You are tasked with generating persuasive text based on the provided context, selected techniques, and application analysis. Your goal is to craft a response that effectively influences the target according to the user's specified goal.

**1. Original User Context:**
{user_context}

**2. Selected Techniques & Details:**
{formatted_details}

**3. Recommended Application Analysis:**
{application_analysis}

**4. User Sentiment**
{sentiment_analysis(query.prompt)}

**5. Task:**
Generate the final persuasive text (e.g., email, chat message, script outline) based *only* on the information provided above.

**General Instructions:**
- Directly apply the insights from the 'Recommended Application Analysis'.
- Incorporate the principles from the 'Selected Techniques & Details'.
- Ensure the output strictly adheres to the 'Output Requirements' (Tone, Length) and especially the 'Ethical Lines NOT to Cross' specified in the 'Original User Context'.
- Do NOT add any extra commentary, preamble, or explanation beyond the requested persuasive text itself.

**Specific Guidance for Salary Negotiation Requests (if applicable):**
- **Be Specific & Evidence-Based:** Instead of vague claims ("hard work"), list 2-3 concrete, quantifiable achievements (e.g., "Led X project resulting in Y outcome," "Implemented Z process saving X hours/dollars," "Exceeded sales target by X%").
- **Professional Tone:** Maintain a respectful, professional tone. Avoid overly casual language ("hustled") or assuming agreement. Focus on facts and results.
- **Request a Meeting, Not a Specific Amount (Initially):** The primary goal of an initial request is often to secure a meeting. Frame the request as wanting to discuss performance and compensation, rather than demanding a specific figure upfront.
- **Link Performance to Request:** Clearly use specific achievements as the justification for requesting the discussion.
- **Frame Around Value Delivered:** Emphasize how your actions benefited the company (increased revenue, saved costs, improved efficiency, etc.), not just "fairness" to you.
- **Strategic Market Research Mention (Optional):** If using market data, mention it as supporting context for the *discussion*, not the sole driver of the request.

**Generate the persuasive text now:**
"""
            logger.debug(
                f"Final generation prompt (approx length {len(final_prompt)}):\n{final_prompt}"
            )  # Added length log

            # Stream the final response
            logger.info(
                "Making LLM call 3 for final response generation..."
            )  # Added log
            final_response_stream = response_handler.create_text_stream(
                "FINAL_RESPONSE"
            )
            logger.info("Streaming final response...")
            async for chunk in self._model_provider.query_stream(final_prompt):
                await final_response_stream.emit_chunk(chunk)

            await final_response_stream.complete()
            await response_handler.complete()
            # Use session.activity_id for logging
            logger.info(
                f"Successfully completed response generation for activity {session.activity_id}"
            )

        except Exception as e:
            # Use session.activity_id for logging
            logger.error(
                f"Error during assist method for activity {session.activity_id}: {e}",
                exc_info=True,
            )
            try:
                await response_handler.emit_text_block(
                    "ERROR", f"An unexpected error occurred: {e}"
                )
                await response_handler.complete()
            except Exception as resp_err:
                logger.error(f"Failed to send error response to client: {resp_err}")


# --- Server Setup (Phase 3) ---
if __name__ == "__main__":
    logger.info("Starting Persuasion Agent server...")
    # Create an instance of the PersuasionAgent
    agent = PersuasionAgent()
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()
