from __future__ import annotations


def upsert_markdown_section(existing: str, marker: str, lines: list[str]) -> str:
    section = "\n".join(lines)
    start = existing.find(marker)
    if start == -1:
        return existing.rstrip() + "\n\n" + section
    next_start = existing.find("\n## ", start + len(marker))
    prefix = existing[:start].rstrip()
    if next_start == -1:
        return prefix + "\n\n" + section
    suffix = existing[next_start:].lstrip("\n")
    return prefix + "\n\n" + section.rstrip() + "\n\n" + suffix
