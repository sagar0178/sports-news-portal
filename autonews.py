"""Fetch, rewrite, and publish football and cricket news for Jekyll."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
HISTORY_FILE = Path("published_history.json")
POSTS_DIRECTORY = Path("_posts")
MODEL_NAME = "gemini-1.5-flash"

RSS_FEEDS: dict[str, str] = {
    "Football": "http://feeds.bbci.co.uk/sport/football/rss.xml",
    "Cricket": "http://feeds.bbci.co.uk/sport/cricket/rss.xml",
}


def load_published_history() -> set[str]:
    """Load previously processed source URLs."""
    if not HISTORY_FILE.exists():
        return set()

    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.error("Could not read %s: %s", HISTORY_FILE, error)
        return set()

    if not isinstance(history, list):
        LOGGER.error("%s must contain a JSON list", HISTORY_FILE)
        return set()
    return {link for link in history if isinstance(link, str)}


def save_published_history(published_links: set[str]) -> None:
    """Save processed source URLs in a stable JSON format."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(sorted(published_links), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        LOGGER.error("Could not save %s: %s", HISTORY_FILE, error)


def slugify(title: str) -> str:
    """Convert a title to a safe lowercase Jekyll filename component."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_title.lower()).strip("-")
    return slug or "article"


def clean_summary(summary: str) -> str:
    """Remove common RSS markup and decode entities."""
    without_tags = re.sub(r"<[^>]+>", " ", summary)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def initialize_gemini() -> bool:
    """Configure Gemini and report whether an API key is available."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        LOGGER.error("GEMINI_API_KEY is not set")
        return False

    try:
        genai.configure(api_key=api_key)
    except (GoogleAPIError, ValueError) as error:
        LOGGER.error("Could not initialize Gemini: %s", error)
        return False
    return True


def rewrite_with_gemini(
    category: str,
    title: str,
    summary: str,
    source_link: str,
) -> str | None:
    """Rewrite one article summary, returning Markdown body or None on failure."""
    prompt = f"""
Rewrite the following sports news summary into an original, concise Markdown
article. Preserve only facts present in the supplied material. Include a clear
opening paragraph, useful H2 or H3 headings, and a short bullet list titled
"Key takeaways". Do not include YAML frontmatter, a title heading, or invented
details. Finish with:
[Source: {source_link}]({source_link})

Category: {category}
Title: {title}
Summary: {summary}
""".strip()

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        body = response.text.strip()
        if not body:
            raise ValueError("Gemini returned an empty response")
        return body
    except (GoogleAPIError, AttributeError, ValueError) as error:
        LOGGER.error("Gemini failed for %r: %s", title, error)
        return None


def make_frontmatter(title: str, category: str, published_at: datetime) -> str:
    """Build the required Jekyll frontmatter without relying on model output."""
    quoted_title = json.dumps(title, ensure_ascii=False)
    date_value = published_at.strftime("%Y-%m-%d %H:%M:%S +0000")
    return (
        "---\n"
        "layout: post\n"
        f"title: {quoted_title}\n"
        f"date: {date_value}\n"
        f"categories: {category}\n"
        "---\n\n"
    )


def save_article(
    category: str,
    title: str,
    source_link: str,
    body: str,
    published_at: datetime,
) -> Path | None:
    """Write one Jekyll post and return its path."""
    os.makedirs("_posts", exist_ok=True)
    filename = (
        f"{published_at.strftime('%Y-%m-%d')}-{slugify(category)}-"
        f"{slugify(title)}.md"
    )
    post_path = POSTS_DIRECTORY / filename

    try:
        content = make_frontmatter(title, category, published_at) + body.rstrip() + "\n"
        post_path.write_text(content, encoding="utf-8")
    except OSError as error:
        LOGGER.error("Could not save %s: %s", post_path, error)
        return None
    return post_path


def entry_details(entry: Any, category: str) -> tuple[str, str, str] | None:
    """Extract the title, summary, and source URL from an RSS entry."""
    link = str(entry.get("link", "")).strip()
    title = str(entry.get("title", "")).strip()
    summary = clean_summary(str(entry.get("summary", "")))
    if not link or not title or not summary:
        LOGGER.warning("Skipping incomplete %s RSS entry", category)
        return None
    return title, summary, link


def run_pipeline() -> tuple[int, int]:
    """Process all unread BBC entries and return published and failed counts."""
    if not initialize_gemini():
        return 0, 0

    published_links = load_published_history()
    published_count = 0
    failed_count = 0

    for category, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
        except (OSError, ValueError) as error:
            LOGGER.error("Could not fetch %s feed: %s", category, error)
            failed_count += 1
            continue

        for entry in getattr(feed, "entries", []):
            details = entry_details(entry, category)
            if details is None:
                failed_count += 1
                continue
            title, summary, source_link = details
            if source_link in published_links:
                continue

            body = rewrite_with_gemini(category, title, summary, source_link)
            if body is None:
                failed_count += 1
                continue

            post_path = save_article(
                category,
                title,
                source_link,
                body,
                datetime.now(timezone.utc),
            )
            if post_path is None:
                failed_count += 1
                continue

            published_links.add(source_link)
            save_published_history(published_links)
            published_count += 1
            LOGGER.info("Published %s", post_path)

    return published_count, failed_count


def main() -> None:
    """Run the news publishing pipeline."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s: %(message)s",
    )
    published_count, failed_count = run_pipeline()
    LOGGER.info(
        "Complete: %d articles published, %d articles failed",
        published_count,
        failed_count,
    )


if __name__ == "__main__":
    main()
