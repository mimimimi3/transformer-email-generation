"""Lightweight per-example analysis helpers for subject-generation outputs."""

import re


def lcs_length(words_a, words_b):
    """Calculates the longest common subsequence length between two word lists."""
    table = [[0] * (len(words_b) + 1) for _ in range(len(words_a) + 1)]

    for i in range(1, len(words_a) + 1):
        for j in range(1, len(words_b) + 1):
            if words_a[i - 1] == words_b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[-1][-1]


def normalize_words(text):
    """Converts text into normalized words for rough error analysis."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def rough_rouge_l_f1(prediction, reference):
    """Computes a rough per-example ROUGE-L-style F1 score."""
    pred_words = normalize_words(prediction)
    ref_words = normalize_words(reference)

    if len(pred_words) == 0 or len(ref_words) == 0:
        return 0.0

    lcs = lcs_length(pred_words, ref_words)
    precision = lcs / len(pred_words)
    recall = lcs / len(ref_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def classify_error(generated_subject, reference_subject, rough_score):
    """Assigns a simple error category to each generated subject line."""
    generated_clean = generated_subject.strip()
    reference_clean = reference_subject.strip()

    if generated_clean == "":
        return "Generated output is empty"

    if generated_clean.lower() == reference_clean.lower():
        return "Generated output is an exact match with the reference text"

    word_count = len(generated_clean.split())

    if word_count <= 1:
        return "Generated output too short"

    if word_count > 12:
        return "Generated output too long"

    if rough_score < 0.15:
        return "Generated output has low overlap with reference"

    return "Generated output needs manual review"
