import json
import logging
from urllib.parse import parse_qs, urlparse

from playwright.async_api import Page, Playwright, TimeoutError

from gfwldata.config.replay import ReplaySettings

logger = logging.getLogger(__name__)


INIT_SCRIPT = """
    window.consoleLogs = [];
    const originalLog = console.log;
    console.log = function() {
        window.consoleLogs.push(Array.from(arguments).join(' '));
        originalLog.apply(console, arguments);
    };
"""


class ReplayExtractor:
    """Extracts replay JSON from duelingbook replays."""

    def __init__(self, config: ReplaySettings, playwright_client: Playwright):
        self.config = config
        self.playwright_client = playwright_client

    async def extract_replay_json(self, replay_url: str) -> dict | None:
        """Extracts replay JSON from the replay url."""

        # Initialize browser and page
        browser = await self.playwright_client.chromium.connect_over_cdp(
            endpoint_url=self.config.SBR_WS_ENDPOINT
        )
        page = await browser.new_page()

        # Inject console logger to capture all logs
        await page.add_init_script(INIT_SCRIPT)

        try:
            logger.info("Navigating to replay URL: %s", replay_url)
            await page.goto(
                url=replay_url,
                timeout=self.config.PAGE_TIMEOUT,
                wait_until="domcontentloaded",
            )

            logger.info("Page loaded, awaiting card element for URL: %s", replay_url)
            await page.locator("div.card").first.wait_for(
                timeout=self.config.PAGE_TIMEOUT
            )

            # Get logs from console logger
            console_logs = await page.evaluate("window.consoleLogs")

            if not console_logs:
                logger.error("No console logs captured for URL: %s", replay_url)
                await self._take_screenshot_of_page(page, replay_url)
                return

            replay_json = self._extract_json_from_logs(console_logs)

            if not replay_json:
                raise ReplayExtractionError("Replay JSON not found in console logs")

            return replay_json

        except TimeoutError:
            logger.error("Timeout while loading replay URL: %s", replay_url)
            await self._take_screenshot_of_page(page, replay_url)
            raise

        except Exception:
            logger.exception(
                "Error during extraction of replay JSON for URL: %s", replay_url
            )
            await self._take_screenshot_of_page(page, replay_url)
            raise

        finally:
            await page.close()
            await browser.close()

    def _extract_json_from_logs(self, console_logs: list[str]) -> dict | None:
        """Extracts replay JSON from a list of console logs."""
        for log in console_logs:
            try:
                data = json.loads(log)
                if isinstance(data, dict) and data.get("conceal") is False:
                    return data
            except json.JSONDecodeError:
                continue

    async def _take_screenshot_of_page(self, page: Page, replay_url: str) -> None:
        """Takes a screenshot of the page at the given replay URL."""
        logger.info("Taking screenshot for debugging, URL: %s", replay_url)
        replay_id = self._get_replay_id(replay_url)
        screenshot_path = (
            f"{self.config.SCREENSHOT_DIR}/{replay_id}_no_replay_found.png"
        )
        await page.screenshot(path=screenshot_path)
        logger.info("Screenshot saved to: %s", screenshot_path)

    @staticmethod
    def _get_replay_id(replay_url: str) -> int:
        """Extracts the replay ID from the replay URL query parameters."""
        parsed_url = urlparse(replay_url)
        params = parse_qs(parsed_url.query)
        return int(params.get("id")[0])


class ReplayExtractionError(Exception):
    """Exception wrapper to trigger tenacity retry mechanism"""

    pass
