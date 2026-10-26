"""Fetch unread football and cricket stories from configured RSS feeds."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import feedparser
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
HISTORY_FILE = Path("published_history.json")
POSTS_DIRECTORY = Path("_posts")
ARTICLES_PER_FEED = 3

RSS_FEEDS: dict[str, dict[str, str]] = {
    "Football": {
        "BBC Sport Football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "ESPN Football": "https://www.espn.com/espn/rss/soccer/news",
    },
    "Cricket": {
        "BBC Sport Cricket": "https://feeds.bbci.co.uk/sport/cricket/rss.xml",
        "ESPN Cricinfo": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
    },
}


def load_environment() -> None:
    """Load local environment variables from .env when it is present."""
    load_dotenv()


def initialize_gemini() -> None:
    """Configure the Gemini client from the environment."""
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)


def rewrite_with_gemini(
    category: str,
    original_title: str,
    original_summary: str,
    source_link: str,
) -> str:
    """Rewrite an article as a publication-ready Jekyll/Hugo Markdown document."""
    initialize_gemini()
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    prompt = f"""
You are an expert sports journalist creating an original article for a static
blog. Rewrite the source details below; do not copy phrases or invent facts.
Keep all factual claims grounded in the supplied title and summary.

Return only one complete Markdown document with:
1. YAML frontmatter delimited by --- lines. Include exactly these fields:
   title, date, categories, tags, meta_description, and slug. Use today's date
   in YYYY-MM-DD format, set categories to [{category!r}], and make tags a
   useful YAML list.
2. A completely rewritten, unique article body with a clear introduction,
   meaningful H2 and H3 headings, and a concise "Key takeaways" bullet list.
3. A final source-credit line containing a Markdown link to the source URL.

Original title: {original_title}
Original summary: {original_summary}
Source URL: {source_link}
Today's date: {date.today().isoformat()}
""".strip()

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        rewritten_article = response.text.strip()
        if not rewritten_article:
            raise RuntimeError("Gemini returned an empty article")
        return rewritten_article
    except (GoogleAPIError, ValueError, AttributeError, RuntimeError) as error:
        LOGGER.error("Gemini article rewrite failed: %s", error)
        raise RuntimeError("Unable to rewrite article with Gemini") from error


def load_published_history(history_file: Path = HISTORY_FILE) -> set[str]:
    """Return previously processed article links.

    A missing history file starts with an empty set. Invalid JSON is treated as
    an error so that a damaged history file cannot silently cause duplicates.
    """
    if not history_file.exists():
        return set()

    try:
        contents = history_file.read_text(encoding="utf-8")
        history = json.loads(contents)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read history file {history_file}") from error

    if not isinstance(history, list) or not all(
        isinstance(link, str) for link in history
    ):
        raise ValueError(f"History file {history_file} must contain a JSON list of links")

    return set(history)


def save_published_history(
    published_links: set[str], history_file: Path = HISTORY_FILE
) -> None:
    """Persist processed article links as a sorted JSON list."""
    try:
        history_file.write_text(
            json.dumps(sorted(published_links), indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"Unable to write history file {history_file}") from error


def _clean_summary(summary: str) -> str:
    """Decode common RSS HTML entities and remove simple markup."""
    return html.unescape(summary).replace("<p>", "").replace("</p>", "").strip()


def slugify(title: str) -> str:
    """Convert an article title into a lowercase, filesystem-safe slug."""
    normalized_title = unicodedata.normalize("NFKD", title)
    ascii_title = normalized_title.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_title.lower()).strip("-")
    return slug or "untitled"


def extract_unread_articles(
    feed: Any,
    source: str,
    category: str,
    published_links: set[str],
    limit: int = ARTICLES_PER_FEED,
) -> list[dict[str, str]]:
    """Extract up to ``limit`` unread articles from one parsed RSS feed."""
    articles: list[dict[str, str]] = []
    for entry in getattr(feed, "entries", []):
        link = str(entry.get("link", "")).strip()
        if not link or link in published_links:
            continue

        articles.append(
            {
                "title": str(entry.get("title", "")).strip(),
                "summary": _clean_summary(str(entry.get("summary", ""))),
                "link": link,
                "category": category,
                "source": source,
            }
        )
        if len(articles) == limit:
            break

    return articles


def fetch_articles(
    feeds: dict[str, dict[str, str]] = RSS_FEEDS,
    history_file: Path = HISTORY_FILE,
) -> list[dict[str, str]]:
    """Fetch and return the top unread articles from every configured feed."""
    published_links = load_published_history(history_file)
    selected_articles: list[dict[str, str]] = []

    for category, category_feeds in feeds.items():
        for source, feed_url in category_feeds.items():
            LOGGER.info("Fetching %s", source)
            parsed_feed = feedparser.parse(feed_url)
            if parsed_feed.bozo and not parsed_feed.entries:
                LOGGER.warning("Could not parse %s: %s", feed_url, parsed_feed.bozo_exception)
                continue

            articles = extract_unread_articles(
                parsed_feed,
                source,
                category,
                published_links,
            )
            selected_articles.extend(articles)
            published_links.update(article["link"] for article in articles)

    return selected_articles


def save_article(
    article: dict[str, str],
    rewritten_markdown: str,
    posts_directory: Path = POSTS_DIRECTORY,
    published_on: date | None = None,
) -> Path:
    """Save one rewritten article and return its generated Markdown path."""
    publication_date = published_on or date.today()
    category_slug = slugify(article["category"])
    title_slug = slugify(article["title"])
    post_path = posts_directory / (
        f"{publication_date.isoformat()}-{category_slug}-{title_slug}.md"
    )

    try:
        posts_directory.mkdir(parents=True, exist_ok=True)
        post_path.write_text(rewritten_markdown.rstrip() + "\n", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Unable to save article to {post_path}") from error

    return post_path


def publish_articles(
    articles: list[dict[str, str]],
    history_file: Path = HISTORY_FILE,
    posts_directory: Path = POSTS_DIRECTORY,
) -> tuple[int, int]:
    """Rewrite and save articles, recording links only after successful saves."""
    published_links = load_published_history(history_file)
    saved_count = 0
    failed_count = 0

    for article in articles:
        try:
            rewritten_markdown = rewrite_with_gemini(
                article["category"],
                article["title"],
                article["summary"],
                article["link"],
            )
            post_path = save_article(article, rewritten_markdown, posts_directory)
            published_links.add(article["link"])
            save_published_history(published_links, history_file)
            saved_count += 1
            LOGGER.info("Published %s", post_path)
        except (RuntimeError, ValueError, KeyError) as error:
            failed_count += 1
            LOGGER.error("Could not publish %s: %s", article.get("title", "article"), error)

    return saved_count, failed_count


def run_pipeline() -> tuple[int, int]:
    """Fetch unread articles, publish them, and return success/failure counts."""
    articles = fetch_articles()
    LOGGER.info("Found %d unread articles", len(articles))
    saved_count, failed_count = publish_articles(articles)
    return saved_count, failed_count


def main() -> None:
    """Run the article pipeline and log summary statistics."""
    load_environment()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s: %(message)s",
    )
    try:
        saved_count, failed_count = run_pipeline()
    except (RuntimeError, ValueError) as error:
        LOGGER.error("Pipeline failed: %s", error)
        raise SystemExit(1) from error

    LOGGER.info(
        "Pipeline complete: %d articles published, %d articles failed",
        saved_count,
        failed_count,
    )


if __name__ == "__main__":
    main()
