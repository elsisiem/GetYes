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

# (dotenv loading as before)
dotenv_path = None
current_dir = os.path.dirname(__file__)
for i in range(3):
    potential_path = os.path.join(current_dir, "../" * i, ".env")
    if os.path.exists(potential_path):
        dotenv_path = potential_path
        break
if dotenv_path:
    print(f"DEBUG: Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, verbose=True)
else:
    print("DEBUG: No .env file found.")

# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s [%(levelname)s] - %(message)s"
)

# --- CONSTANTS ---
MAX_PDFS_TO_PROCESS = 2
MAX_FILE_SIZE_MB = 15
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_PDF_TEXT_LEN = 75000
# --- END CONSTANTS ---


class PersuasionAgent(AbstractAgent):
    """
    Your personal persuasion assistant, GetYes! Powered by Gemini, techniques DB, search, and PDF processing.
    """

    # --- UPDATED AGENT NAME ---
    def __init__(self, name: str = "GetYes Persuasion Agent"):
        # --- END UPDATE ---
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
        # --- UPDATED STRUCTURE HELP TEXT ---
        self.REQUIRED_PROMPT_STRUCTURE = """
**Required Prompt Structure:**

**Goal:** [Clearly state the SPECIFIC action or belief change. E.g., "Get approval for X budget"]

**Target:**
*   Name/Role: [e.g., "Jane Doe, Marketing Manager"]
*   Profile Notes: [e.g., "Analytical, skeptical"]
*   Key Motivation/Value: [e.g., "Saving money", "Efficiency"]
*   Main Pain Point/Objection: [e.g., "Budget constraints", "Not enough time"]
*   Relevant Belief/Attitude: [e.g., "Thinks current system is fine"]

**Relationship:**
*   Your Role to Target: [e.g., "Peer", "Their Manager"]
*   History/Trust Level: [e.g., "New relationship", "Trust high"]

**Context:**
*   Situation: [e.g., "Follow-up email", "Initial request"]
*   Channel: [e.g., "Email", "Chat", "Script outline"]
*   Key Background Info/Constraints: [1-2 essential points. e.g., "Ref Q1 results", "Need response by EOD"]

**Output Requirements:**
*   Desired Tone: [e.g., "Formal", "Empathetic", "Urgent"]
*   Desired Length: [e.g., "Short paragraph", "~300 words"]
*   Technique Focus (Optional): [e.g., "Emphasize Scarcity"]
*   Ethical Lines NOT to Cross: [CRUCIAL: e.g., "No lying", "Avoid aggression"]

**[Optional: Add any other critical constraints OR URLs to relevant PDF files like https://example.com/report.pdf - up to 2 PDFs supported]**
"""
        # --- END UPDATE ---
        logger.info("GetYes Agent initialization complete.")

    def _get_technique_details_string(self, technique_names: List[str]) -> str:
        # (implementation as before)
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
                        f"Instructions Overview:\n{instructions[:500] + '...' if len(instructions) > 500 else instructions}"
                    )
                    found = True
                    break
            if not found:
                details.append(f"**Technique: {name}**\nDetails not found.")
        return "\n\n---\n\n".join(details) if details else "No details found."

    def _extract_pdf_urls(self, text: str) -> List[str]:
        # (implementation as before)
        pdf_urls = re.findall(
            r'https?://[^\s"\'<>]+?\.pdf(?:[?#][^\s"\'<>]*)?', text, re.IGNORECASE
        )
        unique_urls = sorted(list(set(pdf_urls)))
        logger.info(
            f"Found {len(unique_urls)} unique potential PDF URLs: {unique_urls}"
        )
        return unique_urls

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str
    ) -> Tuple[Optional[bytes], Optional[str]]:
        # (implementation as before, including User-Agent)
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
                    error = f"HTTP Error {response.status}: {response.reason}"
                    logger.warning(f"Download failed: {error} ({url})")
                    return None, error
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/pdf" not in content_type:
                    logger.warning(
                        f"Unexpected Content-Type '{content_type}' for {url}. Processing anyway."
                    )
                content_length = response.headers.get("Content-Length")
                if (
                    content_length
                    and int(content_length) > MAX_FILE_SIZE_MB * 1024 * 1024
                ):
                    error = f"Exceeds size limit {MAX_FILE_SIZE_MB}MB (Content-Length)"
                    logger.warning(f"{error} ({url})")
                    return None, error
                content = bytearray()
                total_read = 0
                chunk_size = 8192
                try:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        content.extend(chunk)
                        total_read = len(content)
                        if total_read > MAX_FILE_SIZE_MB * 1024 * 1024:
                            error = f"Exceeds size limit {MAX_FILE_SIZE_MB}MB during download."
                            logger.warning(f"{error} ({url})")
                            await response.release()
                            return None, error
                    logger.info(f"Downloaded {total_read} bytes from {url}")
                    return bytes(content), None
                except aiohttp.ClientPayloadError as e:
                    logger.warning(f"Payload error: {e} ({url})")
                    return None, f"Payload error: {e}"
        except asyncio.TimeoutError:
            error = f"Timeout after {DOWNLOAD_TIMEOUT_SECONDS}s"
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
        # (implementation as before)
        try:
            logger.info(f"Parsing PDF from: {url}")
            reader = PdfReader(io.BytesIO(pdf_content), strict=False)
            if reader.is_encrypted:
                logger.warning(f"Encrypted PDF: {url}")
                return None, "PDF is encrypted"
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
        # (implementation as before)
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
        user_context_original = query.prompt.strip()  # Use stripped version

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
            return  # Stop processing here for /start
        # --- End /start handling ---

        # --- PDF Processing (Now uses user_context_original) ---
        pdf_urls = self._extract_pdf_urls(user_context_original)
        processed_pdfs: Dict[str, str] = {}
        pdf_errors: Dict[str, str] = {}
        formatted_pdf_content = "No PDF files processed or provided."  # Default

        if pdf_urls:
            urls_to_process = pdf_urls[:MAX_PDFS_TO_PROCESS]
            # (PDF processing logic as before...)
            if len(pdf_urls) > MAX_PDFS_TO_PROCESS:
                warn_msg = f"Found {len(pdf_urls)} PDFs. Processing first {MAX_PDFS_TO_PROCESS}."
                logger.warning(f"[{activity_id}] {warn_msg}")
                await response_handler.emit_text_block("STATUS", warn_msg)
            else:
                await response_handler.emit_text_block(
                    "STATUS", f"Processing {len(urls_to_process)} PDF(s)..."
                )
            tasks = []
            async with aiohttp.ClientSession() as http_session:
                for url in urls_to_process:
                    tasks.append(self._process_pdf_url(http_session, url))
                results = await asyncio.gather(*tasks)
            successful_count = 0
            for url, text, error in results:
                if error:
                    pdf_errors[url] = error
                elif text is not None:
                    processed_pdfs[url] = text
                    successful_count += 1
            pdf_sections = []
            if processed_pdfs:
                pdf_sections.append(
                    f"Successfully processed {successful_count} PDF file(s). Content:"
                )
                for url, text in processed_pdfs.items():
                    pdf_sections.append(
                        f"\n**--- Start: {os.path.basename(url)} ---**\n```\n{text}\n```\n**--- End: {os.path.basename(url)} ---**"
                    )
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
                    f"**Note:** Processing failed for PDF URLs: {error_summary}"
                )
            if pdf_sections:
                formatted_pdf_content = "\n".join(pdf_sections)
            status_msg = f"Finished file processing. Success: {successful_count}, Failed: {len(pdf_errors)}."
            await response_handler.emit_text_block("STATUS", status_msg)
        # --- End PDF Processing ---

        try:
            # --- Step 1: Initial Analysis (LLM Call 1 - FLASH) ---
            # (Analysis prompt and logic remain the same)
            await response_handler.emit_text_block(
                "STATUS", "Analyzing context and planning..."
            )
            logger.info(
                f"[{activity_id}] Performing LLM Call 1 (Analysis) using FLASH."
            )
            exact_technique_names = [
                t.get("name") for t in self._techniques_db if t.get("name")
            ]
            exact_names_list_str = "\n".join(
                [f'- "{name}"' for name in exact_technique_names]
            )
            # --- UPDATED PERSONA IN PROMPT ---
            analysis_prompt = f"""
You are GetYes, the persuasion AI assistant. Analyze the 'User Context' provided. Extract key info, identify techniques/search needs, note missing details. **Ignore file URLs during this analysis.**

**Analysis Steps & Output Format:**

1.  Extract Goal.
2.  Identify Missing Info (Optional): List major missing sections (Target, etc.).
3.  Select Techniques (Best Effort): Top 3-5 based on available info.
4.  Identify Information Needs (Best Effort): 2-4 search topics based on available info.
5.  Determine Status: `sufficient` if Goal found, `insufficient` otherwise.
6.  Generate JSON Output (Use EXACT key names):
    *   `status`: "sufficient" or "insufficient"
    *   `extracted_goal`: string
    *   `missing_info_notes`: string (optional)
    *   `selected_techniques`: list of strings
    *   `needed_information_topics`: list of strings

**User Context:**
```
{user_context_original}
```

**(Reference Only) Required Prompt Structure:**
```
{self.REQUIRED_PROMPT_STRUCTURE}
```

**(Reference Only) Available Technique Names:**
```
{exact_names_list_str}
```

**IMPORTANT:** Respond ONLY with the JSON. Use exact keys. Do not analyze URLs here.
"""
            # --- END UPDATE ---
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
            # (Parsing logic as before)
            if analysis_result is None:
                error_msg = "Failed parsing analysis from AI."
                logger.error(
                    f"[{activity_id}] Failed JSON parse (Flash): {llm1_response_str}"
                )
                await self._handle_error(response_handler, error_msg)
                return
            status = analysis_result.get("status") or analysis_result.get(
                "validation_status"
            )
            if status is None:
                logger.error(f"[{activity_id}] Status missing: {analysis_result}")
                await self._handle_error(
                    response_handler, "AI analysis missing 'status'."
                )
                return

            selected_technique_names = []
            needed_information_topics = []
            missing_info_notes = analysis_result.get("missing_info_notes")
            if status == "insufficient":
                extracted_goal = analysis_result.get("extracted_goal", "Goal missing")
                logger.warning(f"[{activity_id}] Status insufficient: {extracted_goal}")
                await self._handle_error(
                    response_handler,
                    "Goal unclear. Please state what you want to achieve.",
                )
                return
            elif status == "sufficient":
                selected_technique_names = analysis_result.get(
                    "selected_techniques", []
                )
                needed_information_topics = analysis_result.get(
                    "needed_information_topics", []
                )
                extracted_goal = analysis_result.get("extracted_goal", "Goal extracted")
                if not isinstance(selected_technique_names, list):
                    raise ValueError("Invalid tech list")
                if not isinstance(needed_information_topics, list):
                    raise ValueError("Invalid info list")
                logger.info(
                    f"[{activity_id}] Analysis OK (Flash). Goal:'{extracted_goal}'. Techs:{selected_technique_names}. Info:{needed_information_topics}. Missing:{missing_info_notes or 'None'}"
                )
                await response_handler.emit_json("ANALYSIS_RESULT", analysis_result)
            else:
                logger.error(f"[{activity_id}] Invalid status '{status}' (Flash)")
                await self._handle_error(
                    response_handler, f"AI returned unknown status: {status}"
                )
                return

            # --- Steps 2, 3, 4 (Search, Technique Details) ---
            # (No change needed here)
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
            technique_details_str = self._get_technique_details_string(
                selected_technique_names
            )
            logger.info(f"[{activity_id}] Retrieved tech details.")

            # --- Step 5: Generate Final Persuasive Output (LLM Call 2 - PRO) ---
            await response_handler.emit_text_block(
                "STATUS", "Crafting final response using Pro model..."
            )
            logger.info(
                f"[{activity_id}] Performing LLM Call 2 (Final Generation) using PRO."
            )

            # --- UPDATED PERSONA IN FINAL PROMPT ---
            final_generation_prompt = f"""
You are GetYes, a master persuasion AI assistant. Generate highly persuasive text based ONLY on the info below to achieve the user's Goal. Leverage techniques, search results, and PDF content. Cite web sources used.

**I. Foundational Information:**

*   **Original User Context:**
    ```
    {user_context_original}
    ```
*   **Selected Persuasion Techniques (Apply Principles):**
    ```
    {technique_details_str}
    ```
*   **Note on Missing User Input (If any):** {missing_info_notes if missing_info_notes else "Sufficient initial details provided."}

**II. Supporting Evidence & External Context:**

*   **Web Search Results (Use facts/data ONLY from here):**
    ```
    {search_results_str}
    ```
*   **Provided PDF Content (Use relevant info as context):**
    ```
    {formatted_pdf_content}
    ```

**III. MANDATORY Generation Instructions:**

1.  **Objective:** Achieve user's 'Goal'.
2.  **Persuasive Power:** Detailed, convincing, manipulative (within ethics). Use techniques, context, search, PDF info.
3.  **Handling Missing Info:** If noted above, make *reasonable assumptions*, stating them implicitly/explicitly (e.g., "Assuming efficiency is key..."). Proceed with best effort.
4.  **Tone & Style:** Default professional/formal unless user specified otherwise. Adhere strictly to user's tone.
5.  **Detail & Length:** Detailed, comprehensive. Aim for user's 'Desired Length'.
6.  **Factual Grounding (CRITICAL):** NO FABRICATION. Base external facts (stats, studies) ONLY on 'Web Search Results'. Use PDF content for context/internal points. If support missing, omit claim or state generally + note lack of data. If PDF processing failed, acknowledge context might be missing.
7.  **Referencing (MANDATORY):** End with `References:`. List ONLY web search URLs *actually used*. Format `* [Title](URL)`. State `References: None` if no web results used/found. Do NOT list PDF URLs here.
8.  **Adherence:** Strictly follow 'Ethical Lines NOT to Cross' & other constraints.
9.  **Output Format:** ONLY persuasive text + `References:` section. No extra commentary.

**Generate the persuasive text now:**
"""
            # --- END UPDATE ---
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
                if isinstance(chunk, str) and chunk.startswith("[ERROR:"):
                    logger.error(
                        f"[{activity_id}] Error in final stream (Pro): {chunk}"
                    )
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
                "Request blocked by content safety filters.",
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
    logger.info("Starting GetYes Persuasion Agent server...")  # Updated name
    try:
        if not dotenv_path:
            logger.warning("No .env file found during startup.")
        agent = PersuasionAgent()
        server = DefaultServer(agent)
        logger.info("Server configured. Running GetYes...")  # Updated name
        server.run()
    except Exception as main_err:
        logger.critical(
            f"FATAL ERROR during agent/server setup: {main_err}", exc_info=True
        )
        print(f"FATAL ERROR: {main_err}")
