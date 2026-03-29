#!/usr/bin/env python3
"""
Part Number Extractor

Reads a CSV containing equipment rows where one column has a text blob
with part numbers mixed in with other information. Uses OpenRouter API
(LLM) to intelligently extract individual part numbers, then outputs a
clean CSV with: equipment_name, equipment_code, part_1, part_2, ...

Usage:
  python extract_parts.py input.csv -o output.csv \
      --name-col "Equipment Name" \
      --code-col "Equipment Code" \
      --parts-col "Parts Description"

  # Or auto-detect columns interactively:
  python extract_parts.py input.csv -o output.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def read_input_csv(filepath):
    """Read the input CSV and return (headers, rows)."""
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def extract_parts_with_llm(text_blob, api_key, model):
    """Use OpenRouter LLM to extract part numbers from a messy text blob."""
    prompt = f"""You are an expert at identifying part numbers in unstructured text.

Extract ALL individual part numbers from the text below. Part numbers can be in
mixed formats: alphanumeric (e.g., ABC-12345), pure digits, codes with dashes or
dots, manufacturer part numbers, etc.

Rules:
- Return ONLY a JSON array of strings, each string being one part number.
- Do NOT include equipment codes that match the pattern "XX - ###" (two letters, space, dash, space, digits) — those are equipment codes, not part numbers.
- Do NOT include general descriptions, model names, or random text — only actual part numbers.
- If no part numbers are found, return an empty array: []
- Strip any surrounding whitespace from each part number.

Text:
---
{text_blob}
---

Return ONLY a JSON array, no markdown fences, no explanation."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        parts = json.loads(content)
        if isinstance(parts, list):
            return [str(p).strip() for p in parts if str(p).strip()]
        return []
    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse error from LLM: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  [!] OpenRouter API error: {e}", file=sys.stderr)
        return []


def extract_parts_with_regex(text_blob):
    """Fallback: use regex heuristics to find part numbers in text."""
    # Common part number patterns (adjust as needed)
    patterns = [
        r'\b[A-Z]{2,5}-\d{3,10}\b',           # ABC-12345
        r'\b\d{2,4}-[A-Z]?\d{3,8}\b',         # 12-34567, 12-A3456
        r'\b[A-Z]\d{4,10}\b',                  # A12345
        r'\b\d{5,12}\b',                        # 1234567 (5-12 digit numbers)
        r'\b[A-Z]{1,3}\d{2,4}[A-Z]\d{2,4}\b', # AB12C34
    ]

    # Exclude equipment codes like "AB - 123"
    equipment_code_pattern = re.compile(r'^[A-Za-z]{2}\s*-\s*\d+$')

    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text_blob):
            val = match.group().strip()
            if not equipment_code_pattern.match(val):
                found.add(val)

    return sorted(found)


def pick_column(headers, description, default_guesses):
    """Interactively pick a column or auto-detect from common names."""
    for guess in default_guesses:
        for h in headers:
            if guess.lower() in h.lower():
                return h
    # If no auto-detect, prompt user
    print(f"\nCould not auto-detect the '{description}' column.")
    print("Available columns:")
    for i, h in enumerate(headers):
        print(f"  [{i}] {h}")
    while True:
        choice = input(f"Enter column number for '{description}': ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(headers):
            return headers[int(choice)]
        print("Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract part numbers from equipment CSV text blobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_parts.py input.csv -o parts_output.csv
  python extract_parts.py input.csv -o parts_output.csv \\
      --name-col "Equipment Name" --code-col "Code" --parts-col "Description"

Environment variables:
  OPENROUTER_API_KEY   API key for OpenRouter (alternative to --api-key)
""",
    )

    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "-o", "--output",
        default="parts_output.csv",
        help="Output CSV file path (default: parts_output.csv)",
    )
    parser.add_argument(
        "--name-col",
        help="Column name for equipment name/description (auto-detected if omitted)",
    )
    parser.add_argument(
        "--code-col",
        help="Column name for equipment code (auto-detected if omitted)",
    )
    parser.add_argument(
        "--parts-col",
        help="Column name containing the text blob with part numbers (auto-detected if omitted)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model to use (default: google/gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )

    args = parser.parse_args()

    # Read input
    headers, rows = read_input_csv(args.input_file)
    if not rows:
        print("No rows found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} rows from {args.input_file}")
    print(f"Columns: {headers}")

    # Resolve column names
    name_col = args.name_col or pick_column(
        headers, "equipment name",
        ["equipment name", "name", "make", "equipment", "description", "model"],
    )
    code_col = args.code_col or pick_column(
        headers, "equipment code",
        ["equipment code", "code", "equip code", "eq code", "asset"],
    )
    parts_col = args.parts_col or pick_column(
        headers, "parts text blob",
        ["parts", "part", "description", "details", "components", "bom", "bill of material"],
    )

    print(f"\nUsing columns:")
    print(f"  Equipment name: '{name_col}'")
    print(f"  Equipment code: '{code_col}'")
    print(f"  Parts text:     '{parts_col}'")

    use_llm = bool(args.api_key)
    if use_llm:
        print(f"  Using OpenRouter model: {args.model}")
    else:
        print("  No API key — using regex fallback for part extraction.")
        print("  For better results, set OPENROUTER_API_KEY or use --api-key.")

    # Process each row
    all_results = []
    max_parts = 0

    for i, row in enumerate(rows):
        equip_name = row.get(name_col, "").strip()
        equip_code = row.get(code_col, "").strip()
        text_blob = row.get(parts_col, "").strip()

        print(f"\n[{i+1}/{len(rows)}] {equip_name} ({equip_code})")

        if not text_blob:
            print("  (empty parts text, skipping)")
            all_results.append({"name": equip_name, "code": equip_code, "parts": []})
            continue

        if use_llm:
            parts = extract_parts_with_llm(text_blob, args.api_key, args.model)
        else:
            parts = extract_parts_with_regex(text_blob)

        print(f"  => Found {len(parts)} part(s): {parts[:5]}{'...' if len(parts) > 5 else ''}")
        max_parts = max(max_parts, len(parts))
        all_results.append({"name": equip_name, "code": equip_code, "parts": parts})

        if i < len(rows) - 1 and use_llm:
            time.sleep(args.delay)

    # Write output CSV
    out_headers = ["equipment_name", "equipment_code"]
    for j in range(1, max_parts + 1):
        out_headers.append(f"part_{j}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_headers)
        for result in all_results:
            row_out = [result["name"], result["code"]]
            row_out.extend(result["parts"])
            # Pad with empty strings if fewer parts than max
            while len(row_out) < len(out_headers):
                row_out.append("")
            writer.writerow(row_out)

    print(f"\n{'='*60}")
    print(f"  Output written to: {args.output}")
    print(f"  Total rows: {len(all_results)}")
    print(f"  Max parts in a single row: {max_parts}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
