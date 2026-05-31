#!/usr/bin/env python3
"""
ModelShow2 Atomic Judge Pipeline
=================================
This script takes model responses + a judge's raw text output and ATOMICALLY:
  1. Anonymizes responses (for reference/audit)
  2. De-anonymizes the judge output
  3. Returns ONLY real model names — never placeholder labels

The orchestrator CANNOT receive raw "Response A/B/C" labels because this script
never emits them. De-anonymization is not a step the orchestrator executes;
it is the only thing this script returns.

Input handling (IMPORTANT — reliability + security):
  Model responses and judge output routinely contain single quotes, backticks,
  `$`, and newlines. Passing them through a shell `echo '...'` breaks the command
  and is an injection vector. Prefer writing the JSON payload to a file and
  reading it here, OR piping it on stdin (the content is never evaluated by a shell).

Phase 1 — Anonymize (before sending to judge):
  python3 judge_pipeline.py --file /tmp/anonymize.json
  # or:  cat /tmp/anonymize.json | python3 judge_pipeline.py

Phase 2 — Finalize (after judge returns):
  python3 judge_pipeline.py --file /tmp/finalize.json
  # or:  cat /tmp/finalize.json | python3 judge_pipeline.py

The "finalize" action replaces "deanonymize" from the old skill.
It returns deanonymized_judge_output and ranked_models_deanonymized.
No intermediate placeholder state is ever returned to the orchestrator.
"""

import json
import re
import secrets
import string
import sys
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Use a cryptographically secure random source for anonymization shuffling.
# This ensures the judge cannot infer model identity from presentation order.
_secure_rng = secrets.SystemRandom()


def read_payload() -> dict:
    """Read the JSON payload from `--file PATH` if given, else from stdin.

    Using a file (or stdin) avoids embedding untrusted model/judge text in a
    shell command line, which is both a reliability hazard (quote breakage) and
    a shell-injection vector. The content here is parsed as data only.
    """
    argv = sys.argv[1:]
    if "--file" in argv:
        idx = argv.index("--file")
        if idx + 1 >= len(argv):
            raise ValueError("--file requires a path argument")
        with open(argv[idx + 1], "r", encoding="utf-8") as f:
            return json.loads(f.read())
    return json.loads(sys.stdin.read())


def generate_mapping(model_names: list, label_style: str = "alphabetic", shuffle: bool = True):
    if shuffle:
        # Use cryptographically secure shuffle — prevents any positional bias
        shuffled = list(model_names)
        _secure_rng.shuffle(shuffled)
    else:
        shuffled = list(model_names)

    # Alphabetic labels only have 26 slots (A–Z). Past that, fall back to
    # numeric labels for the whole batch so we never raise IndexError and never
    # produce ambiguous/colliding placeholders.
    if label_style == "alphabetic" and len(shuffled) > 26:
        label_style = "numeric"

    anon_map = {}      # placeholder → model_name
    reverse_map = {}   # model_name → placeholder

    for i, model in enumerate(shuffled):
        if label_style == "numeric":
            label = f"Candidate {i + 1}"
        else:
            label = f"Response {string.ascii_uppercase[i]}"
        anon_map[label] = model
        reverse_map[model] = label

    return anon_map, reverse_map


def get_blind_responses(responses_by_model: dict, reverse_map: dict) -> dict:
    """Returns placeholder → response_text"""
    return {
        reverse_map[model]: text
        for model, text in responses_by_model.items()
        if model in reverse_map
    }


def deanonymize(judge_output: str, anon_map: dict) -> str:
    """Replace all placeholder labels with real model names (bolded)."""
    result = judge_output
    # Sort descending so longer/later labels replace first (avoids partial overlaps)
    for placeholder in sorted(anon_map.keys(), reverse=True):
        real_model = anon_map[placeholder]
        escaped = re.escape(placeholder)
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        result = pattern.sub(f"**{real_model}**", result)
    return result


def rankings_from_scores(scores: dict, anon_map: dict) -> list:
    """Build a deterministic ranking from an explicit {label: score} map.

    This is the preferred path: when the judge emits a structured `scores`
    object, we never have to parse prose. `scores` may also map real model
    names directly (in case the judge already de-anonymized in its head).
    """
    ranked = []
    seen = set()
    for raw_label, raw_score in scores.items():
        # Match the label to a known placeholder case-insensitively.
        placeholder = next((k for k in anon_map if k.lower() == str(raw_label).lower()), raw_label)
        model = anon_map.get(placeholder)
        if model is None:
            # Maybe the judge keyed by real model name already.
            if raw_label in anon_map.values():
                model = raw_label
                placeholder = next((k for k, v in anon_map.items() if v == raw_label), raw_label)
            else:
                continue
        if placeholder in seen:
            continue
        seen.add(placeholder)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        ranked.append({"placeholder": placeholder, "model": model, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return ranked


def extract_rankings(judge_output: str, anon_map: dict) -> list:
    """Extract ranked model list from judge text, returning real model names.

    Fallback only: used when no structured `scores` object is provided.
    """
    ranked = []
    pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(?:###\s*(?:🥇|🥈|🥉|🏆)?\s*)?(?:\d+(?:st|nd|rd|th)?[.:]?\s+)"
        r"(?:Place[:\s]+|Rank[:\s]+)?"
        r"((?:Response|Candidate|Output)\s+[A-Z0-9]+)"
        r".*?(?:Score[:\s]+|—\s*|:\s*)"
        r"(\d+\.?\d*)\s*/\s*10",
        re.MULTILINE | re.IGNORECASE | re.DOTALL
    )

    seen = set()
    for match in pattern.finditer(judge_output):
        raw = match.group(1).strip()
        normalised = next((k for k in anon_map if k.lower() == raw.lower()), raw)
        if normalised in seen:
            continue
        seen.add(normalised)
        score = float(match.group(2))
        model = anon_map.get(normalised)
        if model:
            ranked.append({"placeholder": normalised, "model": model, "score": score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return ranked


def verify_no_placeholders(text: str) -> list:
    """Return any remaining placeholder-shaped strings (should be empty after deanon)."""
    patterns = [
        r"(?<!\w)Response [A-Z](?!\w)",
        r"(?<!\w)Candidate \d+(?!\w)",
        r"(?<!\w)Output [A-Z](?!\w)",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text))
    return found


def main():
    data = read_payload()
    action = data.get("action")

    # ── Phase 1: Anonymize ────────────────────────────────────────────────────
    if action == "anonymize":
        responses = data["responses"]
        label_style = data.get("label_style", "alphabetic")
        shuffle = data.get("shuffle", True)

        anon_map, reverse_map = generate_mapping(list(responses.keys()), label_style, shuffle)
        blind_responses = get_blind_responses(responses, reverse_map)

        # Return anonymization data — orchestrator stores anon_map for Phase 2
        print(json.dumps({
            "anonymization_map": anon_map,       # placeholder → model_name
            "reverse_map": reverse_map,           # model_name → placeholder
            "blind_responses_for_judge": blind_responses  # placeholder → response_text
        }))

    # ── Phase 2: Finalize (de-anonymize judge output) ─────────────────────────
    elif action == "finalize":
        judge_output = data.get("judge_output", "")
        anon_map = data.get("anonymization_map", {})

        if not anon_map:
            # Fallback: accept reverse_map key too (model→placeholder) and invert
            rm = data.get("reverse_map", {})
            sample_key = next(iter(rm), "")
            if re.match(r"(?:Response|Candidate|Output)\s+[A-Z0-9]+", sample_key, re.IGNORECASE):
                anon_map = rm  # already placeholder→model
            else:
                anon_map = {v: k for k, v in rm.items()}  # invert model→placeholder

        if not anon_map:
            print(json.dumps({"error": "finalize action requires 'anonymization_map'"}))
            sys.exit(1)

        deanon_output = deanonymize(judge_output, anon_map)

        # Prefer an explicit {label: score} map from the judge (deterministic);
        # fall back to scraping the prose only when it is absent.
        scores = data.get("scores")
        if isinstance(scores, dict) and scores:
            ranked = rankings_from_scores(scores, anon_map)
            ranking_source = "structured"
        else:
            ranked = extract_rankings(judge_output, anon_map)
            ranking_source = "regex"

        # Verify — warn if any placeholders escaped
        remaining = verify_no_placeholders(deanon_output)
        deanon_ok = len(remaining) == 0

        print(json.dumps({
            "deanonymized_judge_output": deanon_output,
            "ranked_models_deanonymized": ranked,
            "ranking_source": ranking_source,
            "deanonymization_complete": deanon_ok,
            "remaining_placeholders": remaining  # should be [] always
        }))

    # ── Legacy: deanonymize (backward compat with any tooling) ───────────────
    elif action == "deanonymize":
        # Redirect to finalize logic
        data["action"] = "finalize"
        judge_output = data.get("judge_output", "")
        anon_map = data.get("anonymization_map", data.get("reverse_map", {}))

        sample_key = next(iter(anon_map), "")
        if anon_map and not re.match(r"(?:Response|Candidate|Output)\s+[A-Z0-9]+", sample_key, re.IGNORECASE):
            anon_map = {v: k for k, v in anon_map.items()}

        deanon_output = deanonymize(judge_output, anon_map)
        ranked = extract_rankings(judge_output, anon_map)
        remaining = verify_no_placeholders(deanon_output)

        print(json.dumps({
            "deanonymized_judge_output": deanon_output,
            "ranked_models_deanonymized": ranked,
            "deanonymization_complete": len(remaining) == 0,
            "remaining_placeholders": remaining
        }))

    else:
        print(json.dumps({"error": f"Unknown action '{action}'. Use 'anonymize' or 'finalize'."}))
        sys.exit(1)


if __name__ == "__main__":
    main()
