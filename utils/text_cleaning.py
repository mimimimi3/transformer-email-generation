"""Shared text-cleaning helpers for email body and subject preprocessing."""

import re


def clean_text(text, strip_artifacts=False, preserve_newlines=False):
    """Normalizes email / subject text before prompting or training.

    Args:
        text: Raw text value from the dataset (may be None).
        strip_artifacts: When True, removes AESLC-style <<...>> markers and
            attachment filename lines. Used by the DistilGPT-2 continuation task.
        preserve_newlines: When True, keeps newlines (collapsed runs of blank
            lines). When False, replaces line breaks with spaces for subject
            generation models.
    """
    if text is None:
        return ""

    text = str(text)

    if strip_artifacts:
        text = re.sub(r"<<.*?>>", "", text)
        text = re.sub(
            r"[^\n]*\.(doc|docx|xls|xlsx|pdf|ppt|pptx|csv|zip|jpg|jpeg|gif)",
            "",
            text,
            flags=re.IGNORECASE,
        )

    if preserve_newlines:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
    else:
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)

    return text.strip()
