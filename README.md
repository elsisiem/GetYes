# GetYes: Advanced AI Persuasion Agent

GetYes is a sophisticated AI-powered agent engineered to assist users in crafting highly persuasive communication. It transcends simple text generation by integrating advanced language models (Google Gemini Flash & Pro), a curated database of psychological persuasion techniques, real-time web search capabilities, and contextual document analysis (PDF) to maximize influence and achieve specific user-defined objectives ethically.

This agent acts as a strategic co-pilot, analyzing the nuances of your communication challenge—target audience psychology, relational dynamics, contextual factors—and generates tailored messages designed for maximum impact and resonance.

## Core Capabilities & Innovations

- **Dynamic Dual-Model Strategy:** Leverages Google Gemini's `gemini-2.5-flash-preview-04-17` for rapid initial analysis, context validation, technique selection, and search planning, optimizing for speed and cost-efficiency. It then strategically employs the powerful `gemini-2.5-pro-preview-03-25` for the final, high-fidelity persuasive text generation, ensuring depth, nuance, and sophisticated language.
- **Comprehensive Persuasion Technique Library:** Integrates a structured database (`techniques_database.json`) detailing numerous psychological principles (e.g., Cialdini's principles, framing effects, cognitive biases). The agent analyzes the user's context to select and apply the most relevant techniques.
- **Contextual Document Ingestion (PDF):** Users can provide direct URLs (e.g., from http://catbox.moe/) to relevant PDF documents within their prompt. GetYes downloads (up to 2 files, <15MB each), parses, and incorporates the textual content into the final context provided to the Pro model, enabling richer, document-aware communication strategies.
- **Real-time Data Augmentation (Web Search):** When analysis indicates the need for external validation or data points, GetYes automatically formulates search queries and utilizes the Tavily API to fetch relevant, up-to-date information from the web. This grounds persuasive arguments in verifiable facts.
- **Structured Task Definition:** Employs a required prompt structure to guide users in providing the necessary context (Goal, Target, Relationship, Context, Output Requirements, Ethics), ensuring the agent receives sufficient information for effective strategy formulation.
- **Integrated Ethical Framework:** While designed for influence, GetYes operates strictly within user-defined ethical boundaries specified in the prompt. The system prioritizes responsible usage and requires explicit constraints against misuse.
- **Asynchronous Architecture:** Built using `asyncio`, `aiohttp`, and asynchronous library clients (`google-generativeai`, `tavily-python`) for efficient I/O operations, particularly during file downloads and parallel processing.

## High-Level Workflow

1.  **Input Reception:** Receives user prompt via the `sentient-agent-framework` server interface. Checks for `/start` command.
2.  **PDF Processing (Concurrent):** Extracts `.pdf` URLs from the prompt, downloads files asynchronously (with User-Agent, size/timeout limits), and parses text using `pypdf` (via `run_in_executor`).
3.  **Initial Analysis (Gemini Flash):**
    - Validates the prompt structure (relaxed validation: proceeds if Goal is present).
    - Extracts the core `Goal`.
    - Notes any significant missing context fields.
    - Selects 3-5 relevant persuasion techniques based on available information.
    - Identifies 2-4 necessary web search topics.
    - Returns results as structured JSON.
4.  **Web Search (Tavily - Optional):** If search topics are identified, executes queries via Tavily API.
5.  **Technique Detail Retrieval:** Fetches instructions and keywords for selected techniques from the local database.
6.  **Final Prompt Assembly:** Constructs a comprehensive prompt for the Pro model, including:
    - Original user context.
    - Extracted PDF text content (if any).
    - Web search results (if any).
    - Detailed information on selected techniques.
    - Explicit instructions on generation, tone, ethics, referencing, and handling missing info.
7.  **Persuasive Text Generation (Gemini Pro):** Streams the final, detailed, persuasive message, incorporating all context and adhering to mandatory instructions (including factual grounding and referencing).
8.  **Response Delivery:** Sends the generated text (or error messages) back to the user via the response handler.

## Setup and Installation

**Prerequisites:**

- Python 3.10+
- Access to Google AI API (Gemini Models)
- Tavily Search API Key (Optional, for web search)

**Steps:**

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # or
    .\.venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _(Ensure `requirements.txt` includes `sentient-agent-framework`, `google-generativeai`, `tavily-python`, `python-dotenv`, `aiohttp`, `pypdf`)_

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory (or appropriate location based on your setup) with the following content:

    ```dotenv
    # .env
    GEMINI_API_KEY=YOUR_GOOGLE_AI_API_KEY_HERE
    TAVILY_API_KEY=YOUR_TAVILY_API_KEY_HERE

    # Optional: Override default model names if needed
    # GEMINI_FLASH_MODEL=gemini-1.5-flash-xxxxxx
    # GEMINI_PRO_MODEL=gemini-1.5-pro-xxxxxx
    ```

5.  **Run the Agent Server:**
    ```bash
    python -m persuasion_agent.persuasion_agent
    # Or adjust the path based on your project structure if running from root
    # python src/persuasion_agent/persuasion_agent.py
    ```
    The server will start, and the GetYes agent will be ready to receive requests.

## Usage

Interact with the agent via the interface provided by the `sentient-agent-framework` server.

1.  **Get Instructions:** Send the command `/start` to receive a user-friendly guide on how to structure your requests effectively.
2.  **Submit Persuasion Task:** Provide a detailed prompt following the structure outlined in the `/start` message or the `REQUIRED_PROMPT_STRUCTURE` definition within the agent code. Remember:
    - **Specificity is Key:** Clearly define your Goal, Target, and Context.
    - **Include PDF Links:** Paste direct `.pdf` URLs (e.g., from http://catbox.moe/) if you want the agent to analyze documents.
    - **Define Ethical Boundaries:** Explicitly state what the agent should _not_ do.
    - **More Detail = Better Results:** The richer the context you provide, the more tailored and effective the generated persuasive message will be.

The agent will then process your request through its multi-stage workflow and stream back the final persuasive text.

## Ethical Considerations

GetYes is a powerful tool for influence. This power necessitates responsible usage.

- **User Responsibility:** The user is solely responsible for the ethical application of the generated text and the goals they pursue.
- **Mandatory Boundaries:** Users _must_ define clear ethical lines within their prompts. The agent is instructed to adhere strictly to these boundaries.
- **No Guarantees:** Persuasion depends heavily on factors outside the generated text (e.g., relationship dynamics, unforeseen circumstances). GetYes provides assistance but cannot guarantee outcomes.
- **Transparency:** While the agent employs psychological techniques, its purpose is to assist the user, not to engage in deceptive practices beyond the user's explicit instructions and ethical constraints.

Use GetYes thoughtfully and ethically to enhance communication, not to manipulate harmfully.
