# /persuasion_agent/persuasion_agent.py

import logging
import os
import json
import re
import asyncio
import io
from dotenv import load_dotenv
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple

# Import Provider classes
from .providers.model_provider import GeminiModelProvider
from .providers.search_provider import SearchProvider

# Import Google AI libraries
import google.generativeai as genai
from google.generativeai.types import BlockedPromptException, StopCandidateException

# Import other libraries
import aiohttp
from pypdf import PdfReader

# Import Framework components
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)

# --- dotenv loading section ---
# Ensure indentation is correct here. Pylance might flag the 'break' incorrectly.
dotenv_path = None
current_dir = os.path.dirname(__file__)
for i in range(3):
    potential_path = os.path.join(current_dir, "../" * i, ".env")
    if os.path.exists(potential_path):
        dotenv_path = potential_path
        break  # This break is correctly placed within the loop/if structure
if dotenv_path:
    print(f"DEBUG: Loading .env from: {dotenv_path}")
    load_dotenv(
        dotenv_path=dotenv_path, verbose=True
    )  # Removed load_success assignment as it wasn't used
else:
    print("DEBUG: No .env file found.")
# --- end dotenv loading ---


# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s [%(levelname)s] - %(message)s"
)

# --- CONSTANTS ---
MAX_PDFS_TO_PROCESS = 2
MAX_FILE_SIZE_MB = 15
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_PDF_TEXT_LEN = 100000
# --- END CONSTANTS ---


class PersuasionAgent(AbstractAgent):
    """
    GetYes: Advanced AI Persuasion Strategist. Analyzes, plans, and executes
    influence campaigns using Gemini, techniques, search, and documents.
    """

    def __init__(self, name: str = "GetYes Persuasion Agent"):
        logger.info("Initializing GetYes Persuasion Agent...")
        super().__init__(name)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self._model_provider = GeminiModelProvider(api_key=gemini_api_key)
        logger.info("GeminiModelProvider initialized.")
        search_api_key = os.getenv("TAVILY_API_KEY")
        self._search_provider = (
            SearchProvider(api_key=search_api_key) if search_api_key else None
        )
        if self._search_provider:
            logger.info("SearchProvider initialized.")
        else:
            logger.warning("TAVILY_API_KEY not found. Search disabled.")
        db_path = os.path.join(os.path.dirname(__file__), "techniques_database.json")
        logger.info(f"Loading techniques database from: {db_path}")
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                self._techniques_db: List[Dict[str, Any]] = json.load(f)
            if not isinstance(self._techniques_db, list):
                raise TypeError("DB not a list.")
            logger.info(
                f"Techniques database loaded ({len(self._techniques_db)} techniques)."
            )
        except Exception as e:
            logger.critical(f"Failed to load techniques database: {e}", exc_info=True)
            raise
        self.REQUIRED_PROMPT_STRUCTURE = """
**Required Prompt Structure:**

**Goal:** [Clearly state the SPECIFIC action or belief change. E.g., "Get approval for X budget"]

**Target:**
*   Name/Role: [e.g., "Jane Doe, Marketing Manager"]
*   Profile Notes: [e.g., "Analytical, skeptical, values data"]
*   Key Motivation/Value: [e.g., "Saving money", "Efficiency", "Looking good to boss"]
*   Main Pain Point/Objection: [e.g., "Budget constraints", "Not enough time", "Risk aversion"]
*   Relevant Belief/Attitude: [e.g., "Thinks current system is fine", "Open to new ideas if justified"]

**Relationship:**
*   Your Role to Target: [e.g., "Peer", "Their Manager", "Salesperson"]
*   History/Trust Level: [e.g., "New relationship", "Trust high", "Recent disagreement"]

**Context:**
*   Situation: [e.g., "Follow-up email", "Initial request", "Meeting prep"]
*   Channel: [e.g., "Email", "Chat", "Script outline"]
*   Key Background Info/Constraints: [1-2 essential points. e.g., "Ref Q1 results", "Need response by EOD"]

**Output Requirements:**
*   Desired Tone: [e.g., "Formal", "Empathetic", "Urgent", "Subtly manipulative"]
*   Desired Length: [e.g., "Short paragraph", "~300 words"]
*   Technique Focus (Optional): [e.g., "Emphasize Scarcity"]
*   Ethical Lines NOT to Cross: [CRUCIAL: e.g., "No lying", "Avoid aggression", "Don't exploit X"]

**[Optional: Add URLs to relevant PDF files like https://example.com/report.pdf - up to 2 supported]**
"""
        logger.info("GetYes Agent initialization complete.")

    def _get_technique_details_string(self, technique_names: List[str]) -> str:
        """Formats details of selected techniques for the LLM prompt."""
        details = []
        if not technique_names:
            return "No specific techniques selected."
        for name in technique_names:
            found = False
            for technique in self._techniques_db:
                if technique.get("name") == name:
                    instructions = technique.get("instructions", "N/A")
                    details.append(
                        f"**Technique: {name}**\n"
                        f"Keywords: {', '.join(technique.get('keywords', ['N/A']))}\n"
                        f"Instructions:\n{instructions}\n"  # Keep full instructions for planning
                    )
                    found = True
                    break
            if not found:
                details.append(f"**Technique: {name}**\nDetails not found.")
        return "\n\n---\n\n".join(details) if details else "No details found."

    # (... PDF Helper Methods _extract_pdf_urls, _download_file, _extract_pdf_text, _process_pdf_url remain unchanged ...)
    def _extract_pdf_urls(self, text: str) -> List[str]:
        """Finds potential PDF URLs in text."""
        pdf_urls = re.findall(
            r'https?://[^\s"\'<>]+?\.pdf(?:[?#][^\s"\'<>]*)?', text, re.IGNORECASE
        )
        unique_urls = sorted(list(set(pdf_urls)))
        logger.info(f"Found {len(unique_urls)} unique PDF URLs: {unique_urls}")
        return unique_urls

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Downloads file with User-Agent, timeout, size limit."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            logger.info(f"Attempting download: {url}")
            async with session.get(
                url,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                headers=headers,
                allow_redirects=True,
            ) as response:
                logger.debug(f"Status {response.status} for {url}")
                if response.status >= 400:
                    error = f"HTTP Error {response.status}"
                    logger.warning(f"{error} ({url})")
                    return None, error
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/pdf" not in content_type:
                    logger.warning(
                        f"Unexpected Content-Type '{content_type}' for {url}. Processing anyway."
                    )
                content_length = response.headers.get("Content-Length")
                max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
                if content_length and int(content_length) > max_bytes:
                    error = f"Exceeds size {MAX_FILE_SIZE_MB}MB"
                    logger.warning(f"{error} ({url})")
                    return None, error
                content = bytearray()
                total_read = 0
                chunk_size = 8192
                try:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        content.extend(chunk)
                        total_read = len(content)
                        if total_read > max_bytes:
                            error = f"Exceeds size {MAX_FILE_SIZE_MB}MB"
                            logger.warning(f"{error} ({url})")
                            await response.release()
                            return None, error
                    logger.info(f"Downloaded {total_read} bytes from {url}")
                    return bytes(content), None
                except aiohttp.ClientPayloadError as e:
                    logger.warning(f"Payload error: {e} ({url})")
                    return None, f"Payload error: {e}"
        except asyncio.TimeoutError:
            error = f"Timeout {DOWNLOAD_TIMEOUT_SECONDS}s"
            logger.warning(f"{error} ({url})")
            return None, error
        except aiohttp.ClientError as e:
            error = f"Client/Connection Error: {e}"
            logger.warning(f"{error} ({url})")
            return None, error
        except Exception as e:
            error = f"Unexpected download error: {e}"
            logger.error(f"{error} ({url})", exc_info=True)
            return None, error

    def _extract_pdf_text(
        self, pdf_content: bytes, url: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extracts text from PDF bytes."""
        try:
            logger.info(f"Parsing PDF: {url}")
            reader = PdfReader(io.BytesIO(pdf_content), strict=False)
            if reader.is_encrypted:
                logger.warning(f"Encrypted PDF: {url}")
                return None, "PDF encrypted"
            num_pages = len(reader.pages)
            text = ""
            processed_pages = 0
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        processed_pages += 1
                    else:
                        logger.debug(f"No text page {i+1}/{num_pages} of {url}")
                except Exception as page_err:
                    logger.warning(f"Error page {i+1}/{num_pages} of {url}: {page_err}")
                if len(text) > MAX_PDF_TEXT_LEN:
                    logger.warning(f"Truncating PDF {url} at page {i+1}")
                    break
            if not text:
                logger.warning(f"No text extracted: {url}")
                return None, "No text content found"
            if len(text) > MAX_PDF_TEXT_LEN:
                text = text[:MAX_PDF_TEXT_LEN] + "\n... [Truncated]"
            logger.info(
                f"Extracted ~{len(text)} chars from {processed_pages}/{num_pages} pages: {url}"
            )
            return text.strip(), None
        except Exception as e:
            error = f"PDF parse failed: {e}"
            logger.error(f"Error parsing PDF {url}: {e}", exc_info=True)
            return None, error

    async def _process_pdf_url(
        self, session: aiohttp.ClientSession, url: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Downloads and parses a single PDF URL."""
        pdf_content, download_error = await self._download_file(session, url)
        if download_error:
            return url, None, f"Download Error: {download_error}"
        if pdf_content is None:
            return url, None, "Download Error: Content empty."
        loop = asyncio.get_running_loop()
        try:
            extracted_text, parse_error = await loop.run_in_executor(
                None, self._extract_pdf_text, pdf_content, url
            )
            if parse_error:
                return url, None, f"Parsing Error: {parse_error}"
            return url, extracted_text, None
        except Exception as e:
            logger.error(f"Executor error PDF parse {url}: {e}", exc_info=True)
            return url, None, f"Parsing Error: Unexpected ({e})"

    async def assist(
        self, session: Session, query: Query, response_handler: ResponseHandler
    ):
        activity_id = session.activity_id
        logger.info(f"[{activity_id}] Received query: {query.prompt[:150]}...")
        user_context_original = query.prompt.strip()

        # --- Handle /start command ---
        if user_context_original.lower() == "/start":
            logger.info(f"[{activity_id}] Handling /start command.")
            start_message = """
🎯 Welcome to GetYes! Your AI Persuasion Co-Pilot.

Ready to get more 'Yes' answers? I help craft messages that convince and influence.

How to Use GetYes Effectively:

✅ State Your Goal: What specific outcome do you need? (e.g., "Get budget approved", "Convince client to sign").

✅ Describe the Target: Who are you persuading?
    *   Their Role (e.g., "Manager", "Customer")
    *   Key Motivation (e.g., "Saving money", "Gaining status", "Avoiding risk")
    *   Likely Objection (e.g., "Cost", "Time", "Trust issues")

✅ Provide Context: What's the situation? (e.g., "Follow-up email", "Meeting prep"). Any crucial background?

✅ Define Output: What tone? (e.g., "Formal", "Confident", "Empathetic"). Any ethical lines NOT to cross? (e.g., "No lying").

Boost Your Message:

📄 PDF Context: Need me to read a document?
    1. Upload PDF (up to 2 files, <15MB each) to http://catbox.moe/
    2. Paste the direct `.pdf` link(s) below. I'll analyze it!

🌐 Web Search: I can find supporting facts and data online if needed.

Give me the details. I'll analyze, select proven techniques, and help write a message designed to get results.

Ready to increase your influence? Describe your challenge below. 👇
"""
            start_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            await start_stream.emit_chunk(start_message.strip())
            await start_stream.complete()
            await response_handler.complete()
            logger.info(f"[{activity_id}] Sent /start message.")
            return
        # --- End /start handling ---

        # --- PDF Processing ---
        # (Logic as before)
        pdf_urls = self._extract_pdf_urls(user_context_original)
        processed_pdfs = {}
        pdf_errors = {}
        formatted_pdf_content = "No PDF files processed or provided."
        if pdf_urls:
            urls_to_process = pdf_urls[:MAX_PDFS_TO_PROCESS]
            if len(pdf_urls) > MAX_PDFS_TO_PROCESS:
                warn_msg = f"Found {len(pdf_urls)} PDFs. Processing first {MAX_PDFS_TO_PROCESS}."
                logger.warning(f"[{activity_id}] {warn_msg}")
                await response_handler.emit_text_block("STATUS", warn_msg)
            else:
                await response_handler.emit_text_block(
                    "STATUS", f"Processing {len(urls_to_process)} PDF(s)..."
                )
            tasks = []
            successful_count = 0
            async with aiohttp.ClientSession() as http_session:
                for url in urls_to_process:
                    tasks.append(self._process_pdf_url(http_session, url))
                results = await asyncio.gather(*tasks)
            for url, text, error in results:
                if error:
                    pdf_errors[url] = error
                elif text is not None:
                    processed_pdfs[url] = text
                    successful_count += 1
            pdf_sections = []
            if processed_pdfs:
                pdf_sections.append(
                    f"Successfully processed {successful_count} PDF file(s). Content Summary:"
                )
                for url, text in processed_pdfs.items():
                    pdf_sections.append(
                        f"\n**--- Start Summary: {os.path.basename(url)} ---**\n```\n{text[:1000]}...\n```\n**--- End Summary: {os.path.basename(url)} ---**"
                    )  # Summary for prompt
                pdf_sections.append("\n")
            if pdf_errors:
                error_summary = "; ".join(
                    [
                        f"'{os.path.basename(url)}': {err}"
                        for url, err in pdf_errors.items()
                    ]
                )
                logger.warning(f"[{activity_id}] PDF Errors: {error_summary}")
                pdf_sections.append(
                    f"**Note:** Processing failed for PDFs: {error_summary}"
                )
            if pdf_sections:
                formatted_pdf_content = "\n".join(pdf_sections)
            status_msg = f"Finished file processing. Success: {successful_count}, Failed: {len(pdf_errors)}."
            await response_handler.emit_text_block("STATUS", status_msg)
        # --- End PDF Processing ---

        try:
            # --- Step 1: Enhanced Analysis (LLM Call 1 - FLASH) ---
            await response_handler.emit_text_block(
                "STATUS", "Performing initial analysis..."
            )
            logger.info(
                f"[{activity_id}] Performing LLM Call 1 (Enhanced Analysis) using FLASH."
            )
            exact_technique_names = [
                t.get("name") for t in self._techniques_db if t.get("name")
            ]
            exact_names_list_str = "\n".join(
                [f'- "{name}"' for name in exact_technique_names]
            )
            analysis_prompt = f"""
You are GetYes, the persuasion AI assistant. Analyze the 'User Context'. Extract key info, identify potential psychological drivers/vulnerabilities of the target (based ONLY on user input), select initial techniques, identify search needs, and note missing details. **Ignore file URLs during this analysis step.**

**Analysis Steps & Output Format:**

1.  **Extract Goal:** Identify the primary objective.
2.  **Identify Missing Info (Optional):** List major missing sections (Target profile, Relationship, etc.).
3.  **Infer Psychological Drivers/Vulnerabilities (Best Effort):** Based *only* on the user's description of the Target, list 2-3 potential psychological factors (e.g., "Target seems risk-averse", "Motivated by status", "Vulnerable to FOMO"). If target info missing, state "Cannot infer drivers".
4.  **Select Initial Techniques (Best Effort):** Based on available info, select top 3-5 promising techniques.
5.  **Identify Information Needs (Best Effort):** Based on available info/techniques, identify 2-4 specific search topics.
6.  **Determine Status:** `sufficient` if Goal found, `insufficient` otherwise.
7.  **Generate JSON Output (Use EXACT key names):**
    *   `status`: "sufficient" or "insufficient"
    *   `extracted_goal`: string
    *   `missing_info_notes`: string (optional)
    *   `inferred_target_drivers`: list of strings or string "Cannot infer drivers"
    *   `initial_techniques`: list of strings
    *   `needed_information_topics`: list of strings

**User Context:**
```
{user_context_original}
```

**(Reference Only) Available Technique Names:**
```
{exact_names_list_str}
```

**IMPORTANT:** Respond ONLY with the JSON. Use exact keys. Do not analyze URLs here.
"""
            logger.debug(
                f"[{activity_id}] Analysis Prompt (Flash):\n{analysis_prompt[:500]}..."
            )
            llm1_response_str = await self._model_provider.query(
                model_name=self._model_provider.flash_model_name,
                query=analysis_prompt,
                response_format="json",
            )
            logger.info(
                f"[{activity_id}] Raw LLM Call 1 (Flash) Resp: {llm1_response_str[:500]}..."
            )
            analysis_result = self._model_provider.extract_json_from_response(
                llm1_response_str
            )

            if analysis_result is None:
                raise ValueError(
                    f"Failed to parse JSON analysis from Flash: {llm1_response_str}"
                )
            status = analysis_result.get("status") or analysis_result.get(
                "validation_status"
            )  # Fallback
            if status is None:
                raise ValueError(f"'status' key missing in analysis: {analysis_result}")

            if status == "insufficient":
                goal = analysis_result.get("extracted_goal", "Goal missing")
                logger.warning(f"[{activity_id}] Status insufficient: {goal}")
                await self._handle_error(
                    response_handler, "Goal unclear. Please state your objective."
                )
                return
            elif status != "sufficient":
                raise ValueError(f"Unknown status '{status}' from analysis.")

            extracted_goal = analysis_result.get("extracted_goal", "Goal not specified")
            missing_info_notes = analysis_result.get("missing_info_notes")
            inferred_drivers = analysis_result.get("inferred_target_drivers", [])
            initial_techniques = analysis_result.get(
                "initial_techniques", []
            )  # Get initial list
            needed_information_topics = analysis_result.get(
                "needed_information_topics", []
            )
            if not isinstance(inferred_drivers, (list, str)):
                raise ValueError("Invalid drivers format")
            if not isinstance(initial_techniques, list):
                raise ValueError("Invalid tech list format")
            if not isinstance(needed_information_topics, list):
                raise ValueError("Invalid info list format")

            logger.info(
                f"[{activity_id}] Analysis OK (Flash). Goal:'{extracted_goal}'. Drivers:{inferred_drivers}. Techs:{initial_techniques}. Info:{needed_information_topics}. Missing:{missing_info_notes or 'None'}"
            )
            await response_handler.emit_json("ANALYSIS_RESULT", analysis_result)

            # --- **FIX:** Calculate technique_details_str HERE, before LLM Call 2 ---
            technique_details_str = self._get_technique_details_string(
                initial_techniques
            )
            logger.info(
                f"[{activity_id}] Retrieved technique details for planning/execution."
            )
            # --- End Fix ---

            # --- Step 2: Web Search ---
            # (Logic remains the same)
            search_queries = []
            search_results_str = "No web search performed or needed."
            if needed_information_topics and self._search_provider:
                search_queries = needed_information_topics[:3]
                logger.info(f"[{activity_id}] Search queries: {search_queries}")
                await response_handler.emit_text_block(
                    "STATUS", f"Searching web for: {', '.join(search_queries)}..."
                )
                all_results = []
                try:
                    for sq in search_queries:
                        search_result = await self._search_provider.search(
                            sq, max_results=7
                        )
                        all_results.extend(search_result.get("results", []))
                    if all_results:
                        raw_search_results_for_prompt = all_results[:15]
                        logger.info(
                            f"[{activity_id}] Aggregated {len(raw_search_results_for_prompt)} search results."
                        )
                        formatted = [
                            f"Res {i+1}:\n Title: {r.get('title','N/A')}\n URL: {r.get('url','N/A')}\n Snippet: {r.get('content','N/A')}"
                            for i, r in enumerate(raw_search_results_for_prompt)
                        ]
                        search_results_str = "\n---\n".join(formatted)
                        await response_handler.emit_json(
                            "SOURCES", {"results": raw_search_results_for_prompt}
                        )
                    else:
                        search_results_str = "Web search found no relevant results."
                except Exception as e:
                    logger.error(f"[{activity_id}] Search error: {e}", exc_info=True)
                    search_results_str = f"[Search Error: {e}]"
                    await response_handler.emit_json("SOURCES", {"error": str(e)})

            # --- Step 3: Strategic Planning (LLM Call 2 - FLASH) ---
            await response_handler.emit_text_block(
                "STATUS", "Developing persuasion strategy..."
            )
            logger.info(
                f"[{activity_id}] Performing LLM Call 2 (Strategic Planning) using FLASH."
            )

            # Planning prompt uses technique_details_str calculated above
            planning_prompt = f"""
You are GetYes, the persuasion AI strategist. Based on the initial analysis, available context, search results, and PDF content summary (if any), create a detailed **Persuasion Plan**.

**Inputs:**

1.  **User Context:**
    ```
    {user_context_original}
    ```
2.  **Initial Analysis Findings:**
    *   Extracted Goal: {extracted_goal}
    *   Inferred Target Drivers/Vulnerabilities: {inferred_drivers if isinstance(inferred_drivers, str) else ', '.join(inferred_drivers)}
    *   Initial Techniques Considered: {', '.join(initial_techniques)}
    *   Missing Info Notes: {missing_info_notes or "None"}
3.  **Web Search Results:**
    ```
    {search_results_str}
    ```
4.  **PDF Content Summary:**
    ```
    {formatted_pdf_content}
    ```
5.  **Available Technique Details (Reference):**
    ```
    {technique_details_str}
    ```

**Task: Generate the Persuasion Plan**

Output a structured plan (markdown) covering:
*   **Overall Strategy:** Core persuasive approach.
*   **Key Argument/Narrative:** Main logical/emotional thread.
*   **Targeted Emotion(s):** Emotion(s) to evoke/mitigate.
*   **Technique Sequencing & Application:** Ordered list of chosen techniques (confirm/refine from initial list) with specific application notes for this context (e.g., "Use statistic [X]", "Frame inaction as [Y]").
*   **Subtle Phrasing Tactics:** 2-3 specific phrasing types (e.g., "Rhetorical questions", "Future pacing", "'Because' justifications").
*   **Addressing Objections/Drivers:** How to tackle inferred drivers/objections.
*   **Call to Action:** How to conclude.

**Instructions:** Be specific, actionable, aligned with Goal/Ethics. Output ONLY the plan.
"""
            logger.debug(
                f"[{activity_id}] Planning Prompt (Flash):\n{planning_prompt[:500]}..."
            )
            persuasion_plan_str = await self._model_provider.query(
                model_name=self._model_provider.flash_model_name, query=planning_prompt
            )
            logger.info(
                f"[{activity_id}] Generated Plan (Flash). Length: {len(persuasion_plan_str)}"
            )
            logger.debug(f"[{activity_id}] Persuasion Plan:\n{persuasion_plan_str}")
            await response_handler.emit_json(
                "PERSUASION_PLAN", {"plan": persuasion_plan_str}
            )

            # --- Step 4: Technique Details (already retrieved) ---
            # We use the same technique_details_str calculated after step 1

            # --- Step 5: Final Generation (LLM Call 3 - PRO - Execute Plan) ---
            await response_handler.emit_text_block(
                "STATUS", "Executing strategy & generating final response..."
            )
            logger.info(
                f"[{activity_id}] Performing LLM Call 3 (Execute Plan) using PRO."
            )

            # Final generation prompt uses technique_details_str calculated earlier
            final_generation_prompt = f"""
You are GetYes, a master persuasion AI assistant. Meticulously **execute the provided Persuasion Plan** to generate the final text. Use all context (User Input, PDFs, Search Results) as guided by the plan.

**I. Foundational Information:**

*   **Original User Context:**
    ```
    {user_context_original}
    ```
*   **Note on Missing User Input (If any):** {missing_info_notes if missing_info_notes else "Sufficient initial details provided."}

**II. Supporting Evidence & External Context:**

*   **Web Search Results:**
    ```
    {search_results_str}
    ```
*   **Provided PDF Content Summary:**
    ```
    {formatted_pdf_content}
    ```

**III. The Persuasion Plan to Execute:**
```markdown
{persuasion_plan_str}
```

**IV. Persuasion Technique Details (Reference for Plan Execution):**
```
{technique_details_str}
```

**V. MANDATORY Generation Instructions:**

1.  **EXECUTE THE PLAN:** Follow the 'Persuasion Plan'. Implement its strategy, narrative, emotion, technique sequence/application, and phrasing tactics precisely.
2.  **Integrate Info:** Weave in User Context, PDF Content, Search Results *as directed by the plan*.
3.  **Persuasive Power:** Detailed, convincing, manipulative (within ethics). Use sophisticated language, nuance, subtle plan application.
4.  **Handling Missing Info:** If plan needs missing info, make *reasonable assumptions*, stating them implicitly/explicitly. Proceed with best effort.
5.  **Tone & Style:** Adhere to User Context 'Desired Tone' (or default formal) AND plan's tone guidance.
6.  **Factual Grounding (CRITICAL):** NO FABRICATION. Base external facts ONLY on 'Web Search Results'. Use PDF content as context/internal points per plan. Note lack of data if needed. Acknowledge PDF errors if relevant.
7.  **Referencing (MANDATORY):** End with `References:`. List ONLY web search URLs *actually used*. Format `* [Title](URL)`. State `References: None` if none used/found. Do NOT list PDF URLs.
8.  **Adherence:** Strictly follow 'Ethical Lines NOT to Cross' and the Plan.
9.  **Output Format:** ONLY persuasive text + `References:` section. No commentary.

**Execute the Persuasion Plan and generate the final text now:**
"""
            logger.debug(
                f"[{activity_id}] Final Gen Prompt (Pro) approx length: {len(final_generation_prompt)}"
            )
            logger.debug(
                f"[{activity_id}] Final Gen Prompt (Pro) Start:\n{final_generation_prompt[:500]}..."
            )

            final_response_stream = response_handler.create_text_stream(
                "FINAL_RESPONSE"
            )
            logger.info(f"[{activity_id}] Streaming final response (Pro)...")

            stream_error_occurred = False
            async for chunk in self._model_provider.query_stream(
                model_name=self._model_provider.pro_model_name,
                query=final_generation_prompt,
            ):
                # (Stream processing as before)
                if isinstance(chunk, str) and chunk.startswith("[ERROR:"):
                    logger.error(f"[{activity_id}] Error final stream (Pro): {chunk}")
                    await final_response_stream.emit_chunk(
                        f"\n--- AI Error ---\n{chunk}\n"
                    )
                    stream_error_occurred = True
                elif isinstance(chunk, str):
                    await final_response_stream.emit_chunk(chunk)
                else:
                    logger.warning(
                        f"[{activity_id}] Non-string chunk (Pro): {type(chunk)}"
                    )

            await final_response_stream.complete()
            await response_handler.complete()
            if stream_error_occurred:
                logger.warning(f"[{activity_id}] Completed (Pro), but stream errors.")
            else:
                logger.info(f"[{activity_id}] Successfully completed (Pro).")

        # (Exception handling block remains the same)
        except json.JSONDecodeError as e:
            logger.error(f"[{activity_id}] JSON Parse Error: {e}", exc_info=True)
            await self._handle_error(response_handler, f"Failed parse: {e}")
        except ValueError as e:
            logger.error(f"[{activity_id}] Data Val Error: {e}", exc_info=True)
            await self._handle_error(response_handler, f"Invalid data from AI: {e}")
        except BlockedPromptException as e:
            logger.error(f"[{activity_id}] Prompt Blocked: {e}", exc_info=True)
            await self._handle_error(
                response_handler, f"Blocked by safety filters: {e}"
            )
        except StopCandidateException as e:
            logger.error(f"[{activity_id}] Gen Stopped: {e}", exc_info=True)
            await self._handle_error(
                response_handler, f"AI gen stopped unexpectedly: {e}"
            )
        except genai.errors.GoogleAPIError as e:
            logger.error(f"[{activity_id}] API Error: {e}", exc_info=True)
            await self._handle_error(response_handler, f"Google API error: {e}")
        except Exception as e:
            logger.error(f"[{activity_id}] Unexpected Error: {e}", exc_info=True)
            await self._handle_error(response_handler, f"Unexpected error: {e}")

    async def _handle_error(
        self, response_handler: ResponseHandler, error_message: str
    ):
        # (Implementation as before)
        try:
            error_stream = response_handler.create_text_stream("FINAL_RESPONSE")
            display_error = re.sub(
                r"Request blocked.*",
                "Blocked by content safety filters.",
                error_message,
                flags=re.IGNORECASE,
            )
            display_error = display_error[:1000] + (
                "..." if len(display_error) > 1000 else ""
            )
            await error_stream.emit_chunk(f"Agent Error: {display_error}")
            await error_stream.complete()
            await response_handler.complete()
        except Exception as resp_err:
            logger.error(f"Failed to send error response: {resp_err}")


# --- Server Setup ---
if __name__ == "__main__":
    # (Implementation as before)
    logger.info("Starting GetYes Persuasion Agent server...")
    try:
        if not dotenv_path:
            logger.warning("No .env file found during startup.")
        agent = PersuasionAgent()
        server = DefaultServer(agent)
        logger.info("Server configured. Running GetYes...")
        server.run()
    except Exception as main_err:
        logger.critical(f"FATAL ERROR during setup: {main_err}", exc_info=True)
        print(f"FATAL ERROR: {main_err}")
