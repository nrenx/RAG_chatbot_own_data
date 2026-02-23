#!/usr/bin/env python3
"""
Deep crawler for all *.ipospays.com subdomains.
Discovers every reachable page starting from 3 seed URLs,
follows links across all subdomains, and saves results.

Usage:
    pip install requests beautifulsoup4
    python crawl_ipospays.py
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
import time
import re
import sys

# ── Configuration ──────────────────────────────────────────────
SEED_URLS = [
    "https://releases.ipospays.com/ipospays",
    "https://knowledge.ipospays.com/",
    "https://docs.ipospays.com/payment-terminals-integrations",
]

MAX_DEPTH = 4            # How deep to follow links
MAX_PAGES = 1000         # Safety limit
REQUEST_TIMEOUT = 30     # Seconds per request
DELAY_BETWEEN = 0.5      # Polite delay between requests (seconds)
OUTPUT_FILE = "ipospays_crawl_results.json"
URLS_FILE = "ipospays_all_urls.txt"

# File extensions to skip
SKIP_EXTENSIONS = re.compile(
    r"\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|"
    r"pdf|zip|mp4|mp3|xml|json|rss|atom|gz|tar|dmg|exe|apk|"
    r"webp|avif|map|wasm|bin)$",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def is_ipospays_domain(hostname: str) -> bool:
    """Check if hostname is ipospays.com or any subdomain."""
    return hostname == "ipospays.com" or hostname.endswith(".ipospays.com")


def normalize_url(url: str) -> str:
    """Strip fragment, trailing slash, and sort-normalize the URL."""
    parsed = urlparse(url)
    # Remove fragment and trailing slash from path
    path = parsed.path.rstrip("/") or "/"
    # Reconstruct without fragment and query (keep query for unique pages)
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def extract_text(soup: BeautifulSoup) -> str:
    """Extract readable text from BeautifulSoup, removing boilerplate."""
    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract all valid internal links from the page."""
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        # Skip non-http links
        if href.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)

        try:
            parsed = urlparse(full_url)
        except Exception:
            continue

        # Only follow ipospays domains
        if not is_ipospays_domain(parsed.hostname or ""):
            continue

        # Skip static files
        if SKIP_EXTENSIONS.search(parsed.path):
            continue

        normalized = normalize_url(full_url)
        links.append(normalized)

    return links


def crawl():
    """BFS crawl across all *.ipospays.com subdomains."""
    # Queue holds (url, depth)
    queue = deque()
    visited = set()
    discovered_subdomains = set()
    results = []
    errors = []

    # Seed the queue
    for url in SEED_URLS:
        norm = normalize_url(url)
        if norm not in visited:
            queue.append((norm, 0))
            visited.add(norm)

    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"{'='*60}")
    print(f"  iPOSpays Deep Crawler")
    print(f"  Seeds: {len(SEED_URLS)} | Max depth: {MAX_DEPTH} | Max pages: {MAX_PAGES}")
    print(f"{'='*60}\n")

    page_count = 0

    while queue and page_count < MAX_PAGES:
        url, depth = queue.popleft()
        page_count += 1

        # Track subdomain
        hostname = urlparse(url).hostname
        if hostname:
            discovered_subdomains.add(hostname)

        # Progress
        depth_bar = "  " * depth + "→"
        print(f"[{page_count:4d}/{MAX_PAGES}] D{depth} {depth_bar} {url}")

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            content_type = resp.headers.get("Content-Type", "")

            if "text/html" not in content_type:
                print(f"         ⏭  Skipped (content-type: {content_type[:40]})")
                continue

            resp.raise_for_status()
            html = resp.text

        except requests.RequestException as e:
            error_msg = str(e)[:100]
            print(f"         ❌ Error: {error_msg}")
            errors.append({"url": url, "depth": depth, "error": error_msg})
            continue

        # Parse
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract text
        text = extract_text(soup)

        # Store result
        results.append({
            "url": url,
            "title": title,
            "subdomain": hostname,
            "depth": depth,
            "content_length": len(text),
            "text": text,
        })

        # Discover new links if not at max depth
        if depth < MAX_DEPTH:
            new_links = extract_links(soup, url)
            new_count = 0
            for link in new_links:
                if link not in visited:
                    visited.add(link)
                    queue.append((link, depth + 1))
                    new_count += 1
            if new_count:
                print(f"         🔗 Found {new_count} new links (queue: {len(queue)})")

        # Polite delay
        time.sleep(DELAY_BETWEEN)

    # ── Print Summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"  Pages crawled:    {page_count}")
    print(f"  Content pages:    {len(results)}")
    print(f"  Errors:           {len(errors)}")
    print(f"  URLs discovered:  {len(visited)}")
    print(f"  Queue remaining:  {len(queue)}")
    print(f"\n  Subdomains found ({len(discovered_subdomains)}):")
    for sd in sorted(discovered_subdomains):
        # Count pages per subdomain
        sd_count = sum(1 for r in results if r["subdomain"] == sd)
        print(f"    • {sd}  ({sd_count} pages)")

    # ── Save URLs ──────────────────────────────────────────────
    all_urls = sorted(visited)
    with open(URLS_FILE, "w") as f:
        for url in all_urls:
            f.write(url + "\n")
    print(f"\n  ✅ All {len(all_urls)} URLs saved to: {URLS_FILE}")

    # ── Save Full Results ──────────────────────────────────────
    output = {
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": SEED_URLS,
        "config": {
            "max_depth": MAX_DEPTH,
            "max_pages": MAX_PAGES,
        },
        "summary": {
            "pages_crawled": page_count,
            "content_pages": len(results),
            "errors": len(errors),
            "urls_discovered": len(all_urls),
            "subdomains": sorted(discovered_subdomains),
        },
        "pages": results,
        "errors": errors,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Full results saved to: {OUTPUT_FILE}")

    # ── Quick Data Preview ─────────────────────────────────────
    print(f"\n  Pages by depth:")
    for d in range(MAX_DEPTH + 1):
        count = sum(1 for r in results if r["depth"] == d)
        if count:
            print(f"    Depth {d}: {count} pages")

    print(f"\n  Top 10 pages by content length:")
    top = sorted(results, key=lambda x: x["content_length"], reverse=True)[:10]
    for r in top:
        print(f"    {r['content_length']:>6} chars | {r['url'][:70]}")

    return results, all_urls


if __name__ == "__main__":
    results, urls = crawl()
    print(f"\nDone! Run your n8n deep-crawl workflow or use these results directly.")
