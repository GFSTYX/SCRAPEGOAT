import logging

from httpx import AsyncClient, HTTPError

from gfwldata.config.deck import DeckSettings

logger = logging.getLogger(__name__)


class FLDeckExtractor:
    """Asynchronous FormatLibrary deck extractor"""

    def __init__(self, config: DeckSettings, http_client: AsyncClient):
        self.config = config
        self.http_client = http_client

    async def get_page_of_decks(self, page_num: int) -> list[dict]:
        """Fetches a page of decks from the API."""
        url = f"{self.config.EXTRACTOR_BASE_URL}?page={page_num}&limit=100&sort=publishDate:desc&filter=origin:eq:event,format:eq:Goat"

        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()

        except HTTPError as e:
            logger.error("Failed to fetch decks for page %s: %s", page_num, e)
            raise

    async def get_deck(self, deck_id: int) -> dict:
        """Fetches a single deck by its ID."""
        url = f"{self.config.EXTRACTOR_BASE_URL}/{deck_id}"

        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return response.json()

        except HTTPError as e:
            logger.error("Failed to fetch deck with ID %s: %s", deck_id, e)
            raise
