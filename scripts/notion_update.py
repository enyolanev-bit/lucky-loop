#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
NOTION_VERSION = "2022-06-28"


def page_id_from_url_or_id(value: str) -> str:
    value = value.strip()
    m = re.search(r"([0-9a-fA-F]{32})", value.replace("-", ""))
    if not m:
        raise SystemExit("Could not find a 32-char Notion page id in NOTION_PAGE_ID/URL")
    raw = m.group(1).lower()
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def rich(text: str):
    return [{"type": "text", "text": {"content": text[:2000]}}]


def paragraph(text: str):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich(text)}}


def bullet(text: str):
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich(text)}}


def heading(text: str, level: int = 2):
    typ = f"heading_{level}"
    return {"object": "block", "type": typ, typ: {"rich_text": rich(text)}}


def code_block(text: str, language: str = "plain text"):
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": rich(text[:1900]), "language": language},
    }


def append_blocks(page_id: str, token: str, blocks: list[dict]) -> dict:
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    r = requests.patch(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        json={"children": blocks},
        timeout=30,
    )
    if r.status_code >= 300:
        raise SystemExit(f"Notion API error {r.status_code}: {r.text}")
    return r.json()


def read_report_snippet() -> str:
    report = ROOT / "reports" / "final_report.md"
    if not report.exists():
        return "No final_report.md yet."
    lines = report.read_text(encoding="utf-8").splitlines()
    keep = []
    for line in lines:
        if line.startswith("| run_") or line.startswith("Best run:") or line.startswith("Goal:"):
            keep.append(line)
    return "\n".join(keep[:12]) or "Report exists but no timeline lines were found."


def main() -> None:
    token = os.getenv("NOTION_TOKEN")
    page = os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_PAGE_URL")
    if not token or not page:
        raise SystemExit("Set NOTION_TOKEN and NOTION_PAGE_ID or NOTION_PAGE_URL")
    page_id = page_id_from_url_or_id(page)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = os.getenv("LUCKYWORLD_STATUS", "Working prototype: Qwen-AgentWorld predicts experiments before real sklearn execution; traces and report are generated.")
    next_step = os.getenv("LUCKYWORLD_NEXT", "Add a controlled perturbation scenario: noisy labels / data leakage / timeout trap.")
    blocks = [
        heading(f"Lucky Loop update - {now}", 2),
        paragraph(status),
        bullet("Qwen-AgentWorld-35B-A3B is running on Team Pegasus MI300X through vLLM."),
        bullet("Endpoint: http://YOUR_SIMULATOR_HOST:8000/v1"),
        bullet("CLI loop verified: prediction -> real execution -> comparison -> next decision -> report."),
        bullet("Artifacts: runs/run_001.json ... run_005.json and reports/final_report.md."),
        bullet(f"Next: {next_step}"),
        heading("Latest report extract", 3),
        code_block(read_report_snippet(), "markdown"),
    ]
    res = append_blocks(page_id, token, blocks)
    print(f"Appended {len(res.get('results', []))} blocks to Notion page {page_id}")


if __name__ == "__main__":
    main()
