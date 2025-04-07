import logging
import time

from httpx import Client, HTTPStatusError

from gfwldata.config.discord import DiscordSettings

logger = logging.getLogger(__name__)


class DiscordLeagueDataExtractor:
    def __init__(self, config: DiscordSettings, http_client: Client):
        """Extracts league matches data from discord channels."""
        self.config = config
        self.http_client = http_client

    def get_league_data_messages(
        self, min_message_id: int | None = None, max_message_id: int | None = None
    ) -> list:
        """Fetches league data messages from Discord within a given ID range."""
        REQUEST_URL = self.config.REQUEST_BASE_URL
        offset = 0
        max_offset = 0
        messages = []

        logger.info(
            "Starting league data fetch. min_message_id: %s, max_message_id: %s",
            min_message_id,
            max_message_id,
        )

        # Create request headers
        headers = self._prepare_headers()

        while True:
            try:
                # Create request params
                params = self._prepare_params(offset, min_message_id, max_message_id)
                logger.debug("Fetching page with offset: %d", offset)

                # Make request to discord
                response = self.http_client.get(
                    url=REQUEST_URL, headers=headers, params=params
                )

                response.raise_for_status()
                data = response.json()

            except HTTPStatusError as e:
                logger.error(
                    "HTTP error fetching data from %s (status %d): %s",
                    e.request.url,
                    e.response.status_code,
                    e.response.text,
                )
                return messages

            except Exception:
                logger.exception(
                    "Unexpected error during Discord API request to %s",
                    REQUEST_URL,
                )
                return messages

            # Calculate the maximum offset on the first successful request
            if offset == 0:
                total_results = data.get("total_results", 0)

                # If no results in page, then end the loop
                if total_results == 0:
                    logger.info("No messages found on this page. Exiting.")
                    return messages

                # Calculate max offset using total_results
                max_offset = self._calculate_max_offset(total_results)
                logger.debug("Calculated max offset: %i", max_offset)

            # Collect messages
            messages_in_page = data.get("messages", [])
            logger.debug("Received %d messages in page.", len(messages_in_page))

            if not messages_in_page:
                logger.info("No messages found on this page. Exiting.")
                return messages

            for message_array in messages_in_page:
                message: dict = message_array[0]

                # Skip message if validation fails
                if not self._validate_message(message):
                    logger.debug(
                        "Skipping invalid message: %s", message.get("id", "UNKNOWN")
                    )
                    continue

                messages.append(message)

            # If on final page, then end the loop
            if offset >= max_offset:
                logger.info("Reached max offset. Fetch complete.")
                return messages

            # Throttle requests
            time.sleep(self.config.REQUEST_DELAY)

            # Adjust offset
            offset += self.config.REQUEST_PAGE_SIZE

    def _prepare_headers(self) -> dict:
        """Prepares the request headers."""
        return {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            "Authorization": self.config.HEADER_AUTHORIZATION,
            "User-Agent": self.config.HEADER_USER_AGENT,
            "Referer": self.config.HEADER_REFERER,
            "X-Discord-Locale": "en-US",
            "X-Discord-Timezone": "America/New_York",
            "X-Super-Properties": self.config.HEADER_SUPER_PROPERTIES,
        }

    def _prepare_params(
        self,
        offset: int,
        min_message_id: int | None = None,
        max_message_id: int | None = None,
    ) -> dict:
        """Prepares the request parameters in the "normal" order."""
        params = {
            "channel_id": self.config.PARAM_CHANNEL_ID,
            "author_id": self.config.PARAM_AUTHOR_ID,
        }

        if min_message_id:
            params["min_id"] = min_message_id

        if max_message_id:
            params["max_id"] = max_message_id

        # This is last to maintain "normal" param order
        params["offset"] = offset

        return params

    def _calculate_max_offset(self, total_results: int) -> int:
        """Calculates the maximum offset for pagination based on total results."""
        # Calculate total number of pages with floor rounding to adjust for 0-index
        total_pages = total_results // self.config.REQUEST_PAGE_SIZE

        # Calculate the max offset
        return total_pages * self.config.REQUEST_PAGE_SIZE

    def _validate_message(self, message: dict) -> bool:
        """Validates a single message."""
        embeds = message.get("embeds", [])

        # If no embeds, then there's no data
        if not embeds:
            return False

        return True
