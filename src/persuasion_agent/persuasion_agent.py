import logging
import os
import json
import re  # Added import
from dotenv import load_dotenv
from typing import AsyncIterator, List, Dict, Any
from .sentiment import sentiment_analysis

# Assuming ModelProvider exists or will be created similar to search_agent
# If it's generic enough, we might move it to a shared location later.
# For now, let's assume it's specific to this agent's providers.
# We'll need to create this file/directory structure.
from .providers.model_provider import ModelProvider
from .providers.search_provider import SearchProvider  # Added import

from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)

# Load environment variables from .env file in the parent 'persuasion_agent' directory
dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")  # Adjusted path
# Add logging for debugging .env loading
print(f"DEBUG: Attempting to load .env file from: {dotenv_path}")
load_success = load_dotenv(dotenv_path=dotenv_path, verbose=True)  # Add verbose=True
print(f"DEBUG: load_dotenv success: {load_success}")
# Check if the file actually exists at that path
print(f"DEBUG: Does .env file exist at calculated path? {os.path.exists(dotenv_path)}")


logger = logging.getLogger(__name__)
# Configure logging level (e.g., INFO, DEBUG) - Set to DEBUG for more verbose output
logging.basicConfig(level=logging.DEBUG)


class PersuasionAgent(AbstractAgent):
    def __init__(self, name: str = "Persuasion Agent"):
        print("DEBUG: Entering PersuasionAgent.__init__")
        super().__init__(name)
        print("DEBUG: super().__init__ called.")

        # --- Configuration & Initialization ---
        print("DEBUG: Loading MODEL_API_KEY...")
        # 1. Load Model API Key
        model_api_key = os.getenv("MODEL_API_KEY")
        print(f"DEBUG: MODEL_API_KEY value: '{model_api_key}'")
        if not model_api_key:
            print(
                "ERROR: MODEL_API_KEY environment variable not found."
            )  # Use print before logger might be ready
            logger.error("MODEL_API_KEY environment variable not found.")
            raise ValueError("MODEL_API_KEY is not set in the environment")
        # Assuming ModelProvider needs the API key
        print("DEBUG: Initializing ModelProvider...")
        self._model_provider = ModelProvider(api_key=model_api_key)
        print("DEBUG: ModelProvider initialized.")
        logger.info("ModelProvider initialized.")

        # 3. Load Search API Key and Initialize Provider
        print("DEBUG: Loading TAVILY_API_KEY...")
        # Add logging to check the retrieved value
        retrieved_tavily_key = os.getenv("TAVILY_API_KEY")
        print(
            f"DEBUG: Retrieved TAVILY_API_KEY value: '{retrieved_tavily_key}' (Type: {type(retrieved_tavily_key)})"
        )  # Print value and type
        search_api_key = retrieved_tavily_key  # Use the retrieved value

        if not search_api_key:
            # Allow agent to run without search, but log a warning
            print("DEBUG: TAVILY_API_KEY not found or empty. Disabling search.")
            logger.warning(
                "TAVILY_API_KEY environment variable not found or empty. Search functionality will be disabled."
            )
            self._search_provider = None
        else:
            print("DEBUG: Initializing SearchProvider...")
            self._search_provider = SearchProvider(api_key=search_api_key)
            print("DEBUG: SearchProvider initialized.")
            logger.info("SearchProvider initialized.")

        # 2. Load Techniques Database
        print("DEBUG: Loading techniques database...")
        # Load from the parent 'persuasion_agent' directory
        db_path = os.path.join(
            os.path.dirname(__file__), "../../techniques_database.json"  # Adjusted path
        )
        print(f"DEBUG: Techniques database path: {db_path}")
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self._techniques_db: List[Dict[str, Any]] = json.load(f)
            print("DEBUG: Techniques database loaded successfully.")
            logger.info(f"Techniques database loaded successfully from {db_path}.")
            if not isinstance(self._techniques_db, list):
                print("WARNING: Techniques database is not a list.")
                logger.warning("Techniques database is not a list.")
                self._techniques_db = []  # Ensure it's a list
        except FileNotFoundError:
            print(f"ERROR: Techniques database file not found at {db_path}")
            logger.error(f"Techniques database file not found at {db_path}")
            raise
        except json.JSONDecodeError as json_err:
            print(
                f"ERROR: Error decoding JSON from techniques database at {db_path}: {json_err}"
            )
            logger.error(
                f"Error decoding JSON from techniques database at {db_path}: {json_err}"
            )
            raise
        except Exception as e:
            print(
                f"ERROR: An unexpected error occurred loading techniques database: {e}"
            )
            logger.error(
                f"An unexpected error occurred loading techniques database: {e}"
            )
            raise
        print("DEBUG: Exiting PersuasionAgent.__init__")

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
            # --- Step 1: Integrated Validation & Technique Selection (LLM Call 1) ---
            # (Code from original lines 305-691 will go here, modified later)
            logger.info(
                f"Prompt validation passed for activity {session.activity_id}."
            )  # Placeholder - real logic comes next

            # --- Step 1: Advanced Technique Selection (LLM Call 1) ---
            await response_handler.emit_text_block(
                "STATUS",
                "Analyzing context, selecting techniques, and identifying information needs...",
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

            # Define the required prompt structure for the LLM
            required_structure = """
**Required Prompt Structure:**

**Goal:** [Clearly state the SPECIFIC action or belief change you want from the target. Be precise.]

**Target:**
*   Name/Role: [Target's Name, Role/Position]
*   Profile Notes: [Brief key descriptors: e.g., Analytical, Mid-50s, Skeptical, Values efficiency, Risk-averse]
*   Key Motivation/Value: [What primarily drives them in this context? e.g., Saving money, Gaining status, Helping team, Avoiding mistakes]
*   Main Pain Point/Objection: [What is their biggest frustration related to this, OR why are they likely to say no?]
*   Relevant Belief: [What do they already believe about this topic/you?]

**Relationship:**
*   Your Role to Target: [e.g., Peer, Their Manager, Reporting to them, Client, Friend]
*   History/Trust: [e.g., New relationship, Trust high, Recent disagreement, Long positive history]

**Context:**
*   Situation: [Why are you writing this now? e.g., Follow-up email, Initial request, Responding to complaint]
*   Channel: [e.g., Email, Chat, Phone script outline]
*   Key Background: [1-2 essential points AI must know/reference. e.g., "Ref previous budget discussion," "Must mention competitor X"]

**Output Requirements:**
*   Desired Tone: [e.g., Formal & Authoritative, Friendly & Empathetic, Urgent & Direct, Confident & Respectful]
*   Desired Length: [e.g., Short paragraph, ~150 words, Bullet points]
*   Technique Focus (Optional): [e.g., Emphasize Social Proof, Use Loss Aversion, Avoid Flattery]
*   Ethical Lines NOT to Cross: [CRUCIAL: e.g., No lying about data, Don't mention family, Avoid aggressive language]

**[Optional: Add any other absolutely critical constraint or piece of information here in 1-2 points]**
"""

            selection_prompt = f"""
**Task:** You must first validate the user's prompt based on the required structure, then potentially select techniques and identify information needs.

**1. Validate User Prompt:**
   - Analyze the **User Context** provided below.
   - Compare it against the **Required Prompt Structure**.
   - Determine if the user has provided sufficient detail for ALL mandatory sections (Goal, Target, Relationship, Context, Output Requirements, Ethical Lines NOT to Cross). Check for presence and reasonable detail (more than just keywords). Pay close attention to 'Ethical Lines NOT to Cross'.

**2. Conditional Action:**
   - **IF the User Context is INSUFFICIENT:**
     - **Generate a detailed, self-contained, and self-explanatory refusal message.** This message MUST:
         - Start by acknowledging the user's request (e.g., "I understand you want help with '[user prompt snippet]'...").
         - Clearly explain *why* specific details are crucial for generating a tailored and effective persuasive message (mention the LLM needs the full picture each time).
         - List the *specific sections or sub-sections* that are missing or lack sufficient detail.
         - Provide a **highly instructive template** based on the **Required Prompt Structure**. For each field in the template:
             - Briefly explain *what* information is needed for that field (e.g., "For 'Goal', state the precise action or belief change you want.").
             - If the user provided *some* information for a field, include it.
             - If a field is missing or insufficient, use a placeholder that *explains what to provide* (e.g., `[Please describe the target's personality, values, and potential objections here.]`).
         - Conclude by asking the user to resubmit the request using the completed template.
     - Respond ONLY with the following JSON format:
       ```json
       {{
         "status": "insufficient",
         "message": "[Your generated detailed, self-explanatory refusal message and instructive template here]"
       }}
       ```
     - Do NOT perform steps 3 and 4 below if the prompt is insufficient.

   - **IF the User Context is SUFFICIENT:**
     - Proceed to steps 3 and 4.
     - Respond ONLY with the JSON format specified in step 4.

**3. Select Techniques (Only if prompt is sufficient):**
   - Select the top 1-3 most effective persuasion/manipulation techniques from the **Available Technique Names** list based on the validated **User Context**.

**4. Identify Information Needs (Only if prompt is sufficient):**
   - Identify 1-3 specific types of factual information (data, statistics, findings, case studies) needed from a web search to support the argument, based on the validated **User Context**. Be specific (e.g., "statistics on success rates of X").
   - Respond ONLY with the following JSON format:
     ```json
     {{
       "status": "sufficient",
       "selected_techniques": ["Exact Technique Name 1", ...],
       "needed_information": ["Specific info needed 1", ...]
     }}
     ```

**User Context:**
```
{user_context}
```

**Required Prompt Structure:**
```
{required_structure}
```

**Available Technique Names:**
```
{exact_names_list_str}
```

**Output Instructions:**
- Adhere STRICTLY to the specified JSON output format based on whether the prompt is sufficient or insufficient.
- Do NOT include any other text, explanation, or formatting outside the single JSON object.
- Ensure technique names in the "sufficient" response match the **Available Technique Names** exactly.
"""
            logger.debug(
                f"Technique & Info Needs selection prompt:\n{selection_prompt}"
            )

            # Use non-streaming query for this call
            llm_response_str = await self._model_provider.query(selection_prompt)
            logger.info(
                f"LLM response for technique/info needs selection: '{llm_response_str}'"
            )

            # Parse the JSON response from LLM Call 1 (Validation & Technique/Info Selection)
            selected_technique_names = []
            needed_information_list = []
            try:
                # Attempt to find JSON within potential markdown code blocks
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_str)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # Assume the whole string might be JSON
                    json_str = llm_response_str.strip()

                llm_response_data = json.loads(json_str)

                # Check the status field
                status = llm_response_data.get("status")

                if status == "insufficient":
                    # Prompt is insufficient, extract message and send as FINAL_RESPONSE
                    refusal_message = llm_response_data.get(
                        "message",
                        "Prompt is insufficient, but no specific message was provided.",
                    )
                    logger.warning(
                        f"LLM validation failed for activity {session.activity_id}. Reason: Prompt insufficient."
                    )
                    logger.debug(f"LLM Refusal Message: {refusal_message}")

                    # Send the refusal message as the final response
                    final_response_stream = response_handler.create_text_stream(
                        "FINAL_RESPONSE"
                    )
                    await final_response_stream.emit_chunk(refusal_message)
                    await final_response_stream.complete()
                    await response_handler.complete()
                    logger.info(
                        f"Sent LLM-generated validation error and completed for activity {session.activity_id}"
                    )
                    return  # Stop processing

                elif status == "sufficient":
                    # Prompt is sufficient, extract techniques and info needs
                    selected_technique_names = llm_response_data.get(
                        "selected_techniques", []
                    )
                    needed_information_list = llm_response_data.get(
                        "needed_information", []
                    )

                    # Basic validation of the extracted lists
                    if not isinstance(selected_technique_names, list) or not all(
                        isinstance(i, str) for i in selected_technique_names
                    ):
                        raise ValueError(
                            "Invalid format for selected_techniques in 'sufficient' response"
                        )
                    if not isinstance(needed_information_list, list) or not all(
                        isinstance(i, str) for i in needed_information_list
                    ):
                        raise ValueError(
                            "Invalid format for needed_information in 'sufficient' response"
                        )

                    logger.info(
                        f"LLM validation passed for activity {session.activity_id}."
                    )
                    logger.info(f"Selected techniques: {selected_technique_names}")
                    logger.info(f"Needed information: {needed_information_list}")

                    # Emit inputs for visibility/debugging before proceeding
                    await response_handler.emit_json(
                        "ANALYSIS_INPUTS",
                        {
                            "techniques": selected_technique_names,
                            "needed_info": needed_information_list,
                        },
                    )
                    # --- Proceed to next steps (Search Query Gen, etc.) ---

                else:
                    # Invalid status value
                    raise ValueError(
                        f"Invalid 'status' value received from LLM: {status}"
                    )

            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode JSON response from LLM Call 1: {e}\nResponse: {llm_response_str}"
                )
                await response_handler.emit_text_block(
                    "ERROR", f"Failed to parse LLM response for validation/selection."
                )
                await response_handler.complete()
                return
            except ValueError as e:  # Catches invalid status or list formats
                logger.error(
                    f"Invalid data format in LLM Call 1 JSON response: {e}\nResponse: {llm_response_str}"
                )
                await response_handler.emit_text_block(
                    "ERROR",
                    f"Invalid data format in LLM response for validation/selection.",
                )
                await response_handler.complete()
                return
            except Exception as e:  # Catch other potential errors
                logger.error(
                    f"Unexpected error processing LLM Call 1 response: {e}\nResponse: {llm_response_str}"
                )
                await response_handler.emit_text_block(
                    "ERROR", f"Error processing LLM response for validation/selection."
                )
                await response_handler.complete()
                return

            # --- If status was 'sufficient', execution continues here ---

            # --- Step 1.5: Generate Search Queries (LLM Call 1.5) ---
            search_queries = []
            if (
                needed_information_list and self._search_provider
            ):  # Only generate if needed and provider exists
                await response_handler.emit_text_block(
                    "STATUS", "Generating targeted search queries..."
                )
                logger.info("Making LLM call 1.5 for search query generation...")

                # Format needed info for the prompt
                needed_info_str = "\n".join(
                    [f"- {info}" for info in needed_information_list]
                )

                search_query_generation_prompt = f"""
Given the following descriptions of information needed to support a persuasive argument, generate 1-3 concise and effective search engine queries to find this information.

**Information Needed Descriptions:**
{needed_info_str}

**Instructions:**
- Create queries that are natural and likely to yield factual data, statistics, or specific evidence (e.g., phrase as questions or use relevant keywords).
- **IMPORTANT: Do NOT enclose queries or parts of queries in double quotes (" ") unless searching for a very specific, known, multi-word phrase.** Prefer broader, unquoted terms.
- Generate a maximum of 3 effective queries.
- Respond ONLY with the search queries, one per line. Do not include any other text, explanation, or formatting.
"""
                logger.debug(
                    f"Search query generation prompt (LLM 1.5):\n{search_query_generation_prompt}"
                )

                try:
                    search_queries_str = await self._model_provider.query(
                        search_query_generation_prompt
                    )
                    # Basic validation: non-empty lines, longer than 3 chars
                    search_queries = [
                        q.strip()
                        for q in search_queries_str.splitlines()
                        if q.strip() and len(q.strip()) > 3
                    ]
                    logger.info(f"Generated search queries: {search_queries}")
                except Exception as e:
                    logger.error(
                        f"Error during LLM Call 1.5 (Search Query Generation): {e}"
                    )
                    # Proceed without search queries if generation fails
                    search_queries = []  # Ensure it's an empty list on error

            elif not self._search_provider:
                logger.info(
                    "Search provider not available. Skipping search query generation."
                )
            else:
                logger.info(
                    "No specific information needs identified by LLM 1. Skipping search query generation."
                )

            # --- Step 2: Web Search Execution ---
            aggregated_search_results = []
            # Default value, will be updated after synthesis if search occurs
            search_results_str = "No web search performed or no results found."

            if search_queries:  # Only search if queries were generated successfully
                await response_handler.emit_text_block(
                    "STATUS", "Searching the web for supporting information..."
                )
                logger.info(f"Executing search for queries: {search_queries}")

                for search_query in search_queries[
                    :3
                ]:  # Limit to max 3 queries executed
                    try:
                        # Using include_raw_content=False is more efficient
                        # max_results=5 gives more data for the synthesis step
                        # Increase max_results to get more raw data from Tavily
                        search_result = await self._search_provider.search(
                            search_query  # Removed max_results
                        )
                        # Add logging to inspect the raw result from Tavily
                        logger.info(
                            f"RAW SEARCH RESULT for '{search_query}': {search_result}"
                        )
                        if (
                            search_result
                            and "results" in search_result
                            and search_result["results"]
                        ):
                            aggregated_search_results.extend(search_result["results"])
                        logger.debug(
                            f"Search results for '{search_query}': {search_result}"
                        )
                    except Exception as search_err:
                        logger.error(
                            f"Error during search for query '{search_query}': {search_err}"
                        )
                        # Continue with other queries if one fails

                if aggregated_search_results:
                    # Increase limit for results passed to synthesis step
                    max_results_for_synthesis = (
                        20  # Allow up to 20 results for synthesis step
                    )
                    aggregated_search_results = aggregated_search_results[
                        :max_results_for_synthesis
                    ]
                    logger.info(
                        f"Aggregated {len(aggregated_search_results)} search results for synthesis."
                    )
                    # Emit raw results for visibility/debugging
                    await response_handler.emit_json(
                        "SOURCES", {"results": aggregated_search_results}
                    )
                    # Note: search_results_str is now prepared in the *next* step (Synthesis)
                else:
                    logger.info("No relevant search results found from web search.")
            # If search_queries was empty, aggregated_search_results remains empty

            # --- Step 3: Information Synthesis (LLM Call 2) ---
            information_sheet_str = (
                "No relevant information synthesized from web search."  # Default
            )

            if aggregated_search_results:  # Only synthesize if we have results
                await response_handler.emit_text_block(
                    "STATUS", "Synthesizing relevant information from search results..."
                )
                logger.info("Making LLM call 2 for Information Synthesis...")

                # Prepare raw results string for the prompt
                raw_results_input_str = json.dumps(aggregated_search_results, indent=2)

                synthesis_prompt = f"""
Analyze the following raw web search results in the context of the original user request and the identified information needs. Extract only the most relevant facts, statistics, and specific data points that directly address the information needs and support the user's persuasive goal. Preserve the source URL and Title for each piece of extracted information. Synthesize these key findings into a concise, structured "Information Sheet". Discard irrelevant, redundant, or low-quality results.

**Original User Context:**
{user_context}

**Identified Information Needs:**
{needed_info_str if needed_information_list else "None specified"}

**Raw Web Search Results (JSON format):**
```json
{raw_results_input_str}
```

**Instructions:**
- Focus strictly on extracting factual information relevant to the 'Information Needed Descriptions'.
- For each extracted fact/statistic/data point, clearly associate it with its original source Title and URL from the raw results.
- Structure the output as an "Information Sheet" (e.g., using markdown bullet points or key-value pairs). Example format:
    *   Fact/Statistic 1 (Source: [Title](URL))
    *   Data Point 2 (Source: [Title](URL))
- If no relevant information can be extracted from the provided results, respond ONLY with the text "No relevant information found in search results."
- Do NOT add analysis or commentary, just the extracted information and sources.
"""
                logger.debug(
                    f"Information Synthesis prompt (LLM 2):\n{synthesis_prompt}"
                )

                try:
                    information_sheet_str = await self._model_provider.query(
                        synthesis_prompt
                    )
                    logger.info("LLM Call 2 (Information Synthesis) completed.")
                    logger.debug(
                        f"Synthesized Information Sheet:\n{information_sheet_str}"
                    )
                    # Emit synthesized info for visibility/debugging
                    await response_handler.emit_json(
                        "SYNTHESIZED_INFO", {"info_sheet": information_sheet_str}
                    )
                    # Update search_results_str for the final prompt to use the synthesized version
                    # Keep the original default if synthesis fails or returns the "not found" message
                    if "No relevant information found" not in information_sheet_str:
                        search_results_str = (
                            information_sheet_str  # Use synthesized info
                        )
                    else:
                        search_results_str = (
                            information_sheet_str  # Keep "not found" message
                        )
                        logger.info(
                            "Synthesis resulted in no relevant information found."
                        )

                except Exception as e:
                    logger.error(
                        f"Error during LLM Call 2 (Information Synthesis): {e}"
                    )
                    # Keep the default information_sheet_str
                    information_sheet_str = (
                        "Error synthesizing information from search results."
                    )
                    search_results_str = (
                        information_sheet_str  # Pass error status to final prompt
                    )

            # --- Step 4: Retrieve Technique Details ---
            # (Moved here as it's needed for the final prompt)
            logger.info("Retrieving details for selected techniques...")
            selected_technique_details = []
            if selected_technique_names:  # Check if techniques were actually selected
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
                        logger.warning(
                            f"Details not found for selected technique: {name}"
                        )
            else:
                logger.warning(
                    "No techniques were selected in LLM Call 1. Cannot retrieve details."
                )

            # Format details for the prompt (even if empty)
            formatted_details = "\n\n".join(
                [
                    f"**Technique: {details['name']}**\nInstructions:\n{details['instructions']}\nExample:\n{details['example']}"
                    for details in selected_technique_details
                ]
            )
            if selected_technique_details:
                logger.info("Successfully retrieved and formatted technique details.")
            else:
                logger.info("No technique details to format.")

            # --- Step 5: Final Prompt Generation & Streaming (LLM Call 3) ---
            await response_handler.emit_text_block(
                "STATUS", "Generating final persuasive response..."
            )
            logger.info("Constructing final prompt for persuasive text generation...")
            # Get sentiment analysis result, handle potential None
            sentiment_result = sentiment_analysis(query.prompt)
            sentiment_result_str = (
                str(sentiment_result)
                if sentiment_result is not None
                else "Sentiment analysis unavailable."
            )
            # Use the synthesized 'information_sheet_str' which is now stored in 'search_results_str'
            # Include the formatted technique details in the final prompt
            final_prompt = f"""
You are a master persuasion and manipulation AI. Your task is to generate the most compelling and effective persuasive text possible based on the extensive context provided below. Your primary goal is to achieve the user's specified objective by strategically influencing the target.

**1. Original User Context & Goal:**
{user_context}

**2. Target User Sentiment Analysis:**
{sentiment_result_str}

**3. Selected Persuasion Techniques & Instructions (Apply these strategically):**
{formatted_details if selected_technique_details else "No specific techniques selected or details available. Rely on general persuasion principles and the user context."}

**4. Synthesized Information Sheet (Derived from Web Search - Use for factual grounding):**
{search_results_str}

**5. Core Task:**
Generate the final persuasive text (e.g., email, chat message, script outline) designed to achieve the user's goal stated in the 'Original User Context'.

**General Instructions:**

**1. Maximize Persuasive Impact:**
    - **Leverage Techniques:** Intelligently and subtly weave the principles from the 'Selected Persuasion Techniques & Instructions' into your response. Adapt the examples and instructions to fit the specific context.
    - **Tone & Style:** Adopt the tone specified in the 'Original User Context' under 'Output Requirements'. If no tone is specified, adopt a tone that is most likely to be persuasive for the target described, considering the relationship and context. This might range from empathetic and friendly to authoritative and urgent. Prioritize effectiveness over a default neutral tone.
    - **Emotional Resonance:** Connect with the target on an emotional level where appropriate, using insights from the sentiment analysis and target profile.
    - **Compelling Narrative:** Craft a message that is engaging, clear, and logically flows towards the desired outcome.

**2. Contextual Integration:**
    - **Synthesized Info:** Use facts, statistics, or data points *only* from the 'Synthesized Information Sheet' to support claims where external evidence is needed. Do NOT invent external facts.
    - **User Context:** Ensure the output directly addresses the user's goal, target profile, relationship dynamics, and situational context provided in the 'Original User Context'.

**3. Constraints:**
    - **Ethical Lines:** STRICTLY adhere to the 'Ethical Lines NOT to Cross' specified by the user in the 'Original User Context'. This is non-negotiable.
    - **Output Format:** Adhere to any length or format requirements specified in the 'Original User Context'.
    - **No Extraneous Text:** Generate ONLY the requested persuasive text and the mandatory 'References' section (if applicable). Do not add preamble, self-correction, or explanations about your process.

**4. Factual Grounding & Referencing (For Search Results ONLY):**
    - **Anti-Hallucination (Search Facts):** You are STRICTLY PROHIBITED from inventing, fabricating, or hallucinating ANY data, statistics, studies, or sources *claimed to be from the web search* if they are not explicitly present in the 'Synthesized Information Sheet'. If a claim requires factual support from the web search, you MUST ONLY make that claim if verifiable supporting evidence exists within the 'Synthesized Information Sheet'. If no supporting data is found in the info sheet for a specific claim needing external validation, you MUST either omit the specific claim OR state it generally without specific numbers/attribution AND explicitly mention that supporting data was not found.
    - **References Section:** MANDATORY. Add a 'References' section at the very end. This section MUST list ONLY the sources (URL and Title) associated with the information *you actually used* from the 'Synthesized Information Sheet'. Format each reference precisely as: `* [Title](URL)`. If the 'Synthesized Information Sheet' is empty, indicates no relevant info was found, OR if you did not actually use any information from it, you MUST explicitly state: `References: None`.

**5. Specific Guidance for Salary Negotiation Requests (Apply if relevant):**
    - **Evidence is Key:** Base the request on 2-3 concrete, quantifiable achievements (e.g., "Led X project resulting in Y outcome," "Implemented Z process saving X hours/dollars," "Exceeded sales target by X%"). Use market rate data *from the Information Sheet* if relevant and available, citing the source in References.
    - **Confident & Professional Tone:** Maintain a respectful but confident tone. Focus on facts, results, and value delivered.
    - **Clear Ask (Meeting):** Frame the request as wanting to discuss performance and compensation alignment based on achievements and market value (if applicable).
    - **Value Proposition:** Emphasize how your actions benefited the company (increased revenue, saved costs, improved efficiency).

**Generate the most persuasive and effective text now:** Make it detailed, specific, and professional by default unless stated otherwise. 
"""

            # Inject the sentiment result string into the final prompt
            final_prompt = final_prompt.format(
                user_context=user_context,
                sentiment_result_str=sentiment_result_str,  # Use the formatted variable
                formatted_details=(
                    formatted_details
                    if selected_technique_details
                    else "No specific techniques selected or details available. Rely on general persuasion principles and the user context."
                ),
                search_results_str=search_results_str,
            )

            logger.debug(
                f"Final generation prompt (LLM 3 - approx length {len(final_prompt)}):\n{final_prompt}"
            )

            # Stream the final response
            logger.info(
                "Making LLM call 3 for final response generation..."
            )  # Added log
            final_response_stream = response_handler.create_text_stream(
                "FINAL_RESPONSE"
            )
            logger.info("Streaming final response...")
            # Use the new Gemini streaming method for the final call
            async for chunk in self._model_provider.query_gemini_stream(final_prompt):
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
    print("DEBUG: Entering __main__ block.")
    # Configure logging level (e.g., INFO, DEBUG) - Moved basicConfig earlier
    # logging.basicConfig(level=logging.DEBUG) # Already configured globally
    logger.info("Starting Persuasion Agent server...")
    print("DEBUG: Instantiating PersuasionAgent...")
    # Create an instance of the PersuasionAgent
    try:
        agent = PersuasionAgent()
        print("DEBUG: PersuasionAgent instantiated successfully.")
        # Create a server to handle requests to the agent
        print("DEBUG: Creating DefaultServer...")
        server = DefaultServer(agent)
        print("DEBUG: DefaultServer created.")
        # Run the server
        print("DEBUG: Running server...")
        server.run(port=os.environ.get("PORT", 8080))
    except Exception as main_err:
        print(f"FATAL ERROR during agent/server setup: {main_err}")
        logger.critical(
            f"FATAL ERROR during agent/server setup: {main_err}", exc_info=True
        )
