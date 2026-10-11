"""transcript_refiner.py — Backwards-compatible refinement shim for VoxFlow.

Since v3.0.0 all refinement runs 100% on-device via ``local_cleanup_service``.
This module keeps the historical ``llm_refine`` / ``TRANSCRIPT_CLEANUP_PROMPT``
API so older imports keep working. An optional injectable ``_client`` (any
object exposing ``generate_content(prompt)``) exists purely as a test hook;
no cloud SDK is imported or required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

TRANSCRIPT_CLEANUP_PROMPT: str = """\
# Role & Objective
You are an expert transcript cleanup and text refinement engine. Your task is \
to take raw, unedited, spoken-word transcripts (often containing rambling, \
filler words, stuttering, self-corrections, and tangential thoughts) and \
rewrite them into crisp, coherent, well-structured text while 100% preserving \
the user's original intent, core details, tone, and technical terminology.

# Transformation Rules
1. **Eliminate Filler & False Starts:** Strip all verbal fillers ("um", "uh", \
"like", "you know", "basically", "sort of", "actually", "I mean") and vocal \
hesitations.
2. **Handle Self-Corrections:** When the speaker changes their mind \
mid-sentence (e.g., *"We should meet at 3 PM... no wait, make it 4 PM"*), \
keep only the final intended correction (*"We should meet at 4 PM"*).
3. **De-duplicate & Condense:** Merge repetitive loops, circular explanations, \
and restated ideas into single, clear sentences without losing factual detail.
4. **Preserve Voice & Jargon:** Do not over-formalize casual notes unless the \
input context demands it. Retain exact variable names, tech stacks, code \
references, URLs, numbers, and proper nouns verbatim.
5. **Format for Readability:**
   - Fix grammar, punctuation, and capitalization.
   - Break run-on sentences into natural, readable clauses.
   - If the speech describes a list of tasks, steps, or multi-point thoughts, \
structure them using clean Markdown bullet points or numbered lists.
6. **Zero Hallucination / Zero Extrapolation:** Do not invent facts, fill gaps \
with assumptions, or add polite conversational preambles \
("Here is your text:"). Output ONLY the refined text.

# Few-Shot Examples

## Example 1 (Casual / Thought Stream)
**Input:**
"So uh basically what I was thinking is that we need to like check the API \
response time because right now it's taking like two seconds or three seconds, \
no actually it's closer to 2.4 seconds, and users are going to hate that. We \
should cache the user profile endpoint in Redis, you know?"

**Output:**
We need to optimize the API response time, which is currently averaging around \
2.4 seconds. We should cache the user profile endpoint in Redis to improve \
performance.

---

## Example 2 (Task List / Action Items)
**Input:**
"Okay so first thing we gotta do tomorrow is fix the Docker build error in CI, \
then after that ping Sarah about the Figma mockups... wait before that check \
if the staging database migrated properly, then do the Figma thing, and \
lastly push the hotfix branch."

**Output:**
Tomorrow's action items:
1. Fix the Docker build error in CI.
2. Verify that the staging database migrated properly.
3. Ping Sarah regarding the Figma mockups.
4. Deploy the hotfix branch.

---

# Input Data
Raw Transcript:
{raw_transcript}

# Refined Output:\
"""


# ── LLM client (lazy-initialised) ─────────────────────────────────────────────

_client: Any = None   # optional injected client (test hook); None => local pipeline


def _get_client():
    """Return mock client if set by tests, or None for 100% on-device local execution."""
    global _client
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

async def llm_refine(text: str) -> str:
    """Run transcript through 100% on-device local refinement.

    Maintains backwards-compatibility with test mocks while ensuring
    zero cloud API dependency in production.
    """
    if not text or not text.strip():
        return text

    global _client
    if _client is not None:
        try:
            prompt = TRANSCRIPT_CLEANUP_PROMPT.format(raw_transcript=text)
            response = _client.generate_content(prompt)
            refined = (getattr(response, "text", "") or "").strip()
            return refined if refined else text
        except Exception as exc:
            log.warning("Mock LLM refinement failed (%s); using baseline text.", exc)
            return text

    # Production path: 100% On-Device Local Refinement
    from local_cleanup_service import local_cleanup_service
    result = await local_cleanup_service.refine(text)
    return result.refined_text
