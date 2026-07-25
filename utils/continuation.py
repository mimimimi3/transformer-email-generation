"""Helpers for DistilGPT-2 email continuation prompting and decoding."""

CONTINUATION_MARKER = "\nContinuation:"
EMAIL_MARKER = "Email:"
SUBJECT_MARKER = "Subject:"

# Supported prompt conditions for body-only vs subject + body comparison.
CONDITIONS = ("body_only", "subject_and_body")


def split_email_body(email_body):
    """Split a cleaned email into an incomplete draft and a held-out continuation.

    The split is deterministic (character midpoint) so train / validation / test
    examples are reproducible. Prefer a nearby whitespace boundary so we do not
    cut through the middle of a word when possible.
    """
    midpoint = len(email_body) // 2
    if midpoint <= 0 or midpoint >= len(email_body):
        return email_body, ""

    # Search a small window around the midpoint for a cleaner split point.
    window_start = max(0, midpoint - 40)
    window_end = min(len(email_body), midpoint + 40)
    window = email_body[window_start:window_end]

    whitespace_offsets = [index for index, char in enumerate(window) if char.isspace()]
    if whitespace_offsets:
        best_local = min(
            whitespace_offsets,
            key=lambda index: abs((window_start + index) - midpoint),
        )
        # +1 keeps the whitespace with the prompt side of the split.
        split_at = window_start + best_local + 1
    else:
        split_at = midpoint

    prompt = email_body[:split_at].strip()
    continuation = email_body[split_at:].strip()
    return prompt, continuation


def build_prompt_text(condition, subject, email_prompt):
    """Formats the incomplete email into the condition-specific model prompt."""
    if condition == "body_only":
        return f"{EMAIL_MARKER}\n{email_prompt}{CONTINUATION_MARKER}"

    if condition == "subject_and_body":
        # Subject context is the only difference between the two experiment arms.
        return (
            f"{SUBJECT_MARKER}\n{subject}\n"
            f"{EMAIL_MARKER}\n{email_prompt}{CONTINUATION_MARKER}"
        )

    raise ValueError(f"Unknown condition: {condition}")


def extract_generated_continuation(decoded_text, prompt_text):
    """Removes the prompt from decoded generation so only new text remains."""
    text = decoded_text.strip()
    prompt = prompt_text.strip()

    if text.startswith(prompt):
        continuation = text[len(prompt) :].strip()
    elif CONTINUATION_MARKER in text:
        continuation = text.split(CONTINUATION_MARKER, maxsplit=1)[1].strip()
    else:
        continuation = text

    # If the model starts inventing a new Email/Subject section, cut it off.
    for marker in (EMAIL_MARKER, SUBJECT_MARKER, CONTINUATION_MARKER.strip()):
        if marker in continuation:
            continuation = continuation.split(marker, maxsplit=1)[0].strip()

    return continuation
