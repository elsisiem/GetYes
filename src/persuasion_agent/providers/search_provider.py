# /persuasion_agent/providers/search_provider.py

from tavily import AsyncTavilyClient
import logging

logger = logging.getLogger(__name__)


class SearchProvider:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Tavily API key is required.")
        self.client = AsyncTavilyClient(api_key=api_key)
        logger.info("Tavily SearchProvider initialized.")

    async def search(self, query: str, **kwargs) -> dict:
        """Performs a search using the Tavily client."""
        logger.debug(
            f"Performing Tavily search for query: '{query}' with options: {kwargs}"
        )
        try:
            # Example: Allow passing max_results, include_raw_content etc.
            # Default to reasonable values if not provided
            search_depth = kwargs.get("search_depth", "basic")  # 'basic' or 'advanced'
            max_results = kwargs.get("max_results", 5)
            include_answer = kwargs.get("include_answer", False)
            include_images = kwargs.get("include_images", False)
            include_raw_content = kwargs.get("include_raw_content", False)

            results = await self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_images=include_images,
                include_raw_content=include_raw_content,
                # Add other parameters as needed based on Tavily documentation
            )
            logger.debug(
                f"Tavily search successful for query: '{query}'. Results count: {len(results.get('results', []))}"
            )
            return results
        except Exception as e:
            logger.error(
                f"Tavily search failed for query '{query}': {e}", exc_info=True
            )
            # Return an empty dict or re-raise depending on desired error handling
            return {"error": str(e), "results": []}
