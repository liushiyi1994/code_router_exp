from __future__ import annotations

from routecode.reporting import upsert_markdown_section


def test_upsert_markdown_section_replaces_target_and_preserves_following_section():
    existing = "\n".join(
        [
            "# Demo",
            "",
            "## Target",
            "",
            "old",
            "",
            "## Later",
            "",
            "keep",
        ]
    )

    updated = upsert_markdown_section(existing, "## Target", ["## Target", "", "new"])

    assert "old" not in updated
    assert "new" in updated
    assert "## Later" in updated
    assert "keep" in updated


def test_upsert_markdown_section_appends_missing_section():
    updated = upsert_markdown_section("# Demo\n", "## Added", ["## Added", "", "content"])

    assert updated.endswith("## Added\n\ncontent")
