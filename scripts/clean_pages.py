from __future__ import annotations

import re
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup


RAW_DIR = Path("knowledge_base/raw_html")
OUT_DIR = Path("knowledge_base/clean_text")


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "doc"


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def normalize_text(text: str) -> str:
    # Collapse multiple blank lines and extra spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove very short noise lines (optional; keep conservative)
    lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            lines.append("")
            continue
        # Drop typical UI crumbs / single-word nav leftovers
        if l.lower() in {"home", "characters", "places", "things", "spells"}:
            continue
        lines.append(l)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_main_container(soup: BeautifulSoup):
    """
    HP Lexicon pages in your samples have:
      <section class="article-content content">
        <div class="article-body"> ... main content ... </div>
      </section>
    See examples where <section id="content"> contains <section class="article-content content">. 
    """
    container = soup.select_one("section.article-content .article-body")
    if container:
        return container

    # Fallbacks
    container = soup.find("article")
    if container:
        return container
    return soup.body or soup


def remove_noise(container):
    """
    Remove common non-content blocks found in your sample pages:
    - From the Web block
    - Comments (Pensieve)
    - Nav/breadcrumbs/share widgets
    - Scripts/styles/noscript/svg
    """
    # Always remove these tags wherever they are inside container
    for tag in container.find_all(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Remove known noisy sections/classes/ids
    selectors = [
        "div.from_the_web",        # "From the Web" block (seen in Voldemort sample)
        "section.comments",        # comments block (Pensieve)
        "#comments_block",
        "#disqus_thread",
        "div.breadcrumbs",
        "nav#breadcrumbs",
        "div#share",
        "form",
        "aside",
        "footer",
        "nav",
    ]
    for sel in selectors:
        for el in container.select(sel):
            el.decompose()

    # Remove ad placeholders (e.g., ezoic)
    for el in container.select("[id*='ezoic'], [class*='ezoic'], div[id*='ad-placeholder']"):
        el.decompose()


def html_to_markdown_like_text(container: BeautifulSoup) -> str:
    """
    Keep structure: headings and paragraphs.
    (Not a full HTML->MD converter; just enough for RAG-friendly text.)
    """
    parts: list[str] = []

    # Preserve order by iterating through relevant tags
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"], recursive=True):
        text = node.get_text(" ", strip=True)
        if not text:
            continue

        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4"}:
            level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[name]
            parts.append(f"{level} {text}")
        elif name == "li":
            parts.append(f"- {text}")
        elif name == "blockquote":
            parts.append(f"> {text}")
        else:
            parts.append(text)

    return normalize_text("\n\n".join(parts))


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw dir not found: {RAW_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html_files = sorted(RAW_DIR.glob("*.html"))
    if not html_files:
        raise SystemExit(f"No .html files in {RAW_DIR.resolve()}")

    ok = 0
    for p in html_files:
        html = p.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        title = (soup.title.get_text(" ", strip=True) if soup.title else p.stem).strip()
        title = re.sub(r"\s+\u2013\s+Harry Potter Lexicon\s*$", "", title)  # " – Harry Potter Lexicon"

        container = extract_main_container(soup)
        remove_noise(container)
        body = html_to_markdown_like_text(container)

        # If extraction failed and body is tiny, fallback to container.get_text()
        if len(body) < 400:
            body = normalize_text(container.get_text("\n", strip=True))

        out_name = f"{slugify(title)[:80]}__{stable_id(p.name)}.md"
        out_path = OUT_DIR / out_name

        out_path.write_text(f"# {title}\n\nSourceFile: {p.name}\n\n---\n\n{body}\n", encoding="utf-8")
        ok += 1

    print(f"Done. Cleaned: {ok}. Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
