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


def extract_parts_with_llm(text_blob, api_key, model, sample_parts="", sample_noise=""):
    """Use OpenRouter LLM to extract part numbers from a messy text blob."""
    # Build dynamic sections from user-provided samples
    parts_examples = ""
    if sample_parts:
        parts_examples = f"""
The user has confirmed these are examples of REAL part numbers from their data:
  {sample_parts}
Use these as a guide for the format and style of part numbers to look for."""

    noise_examples = ""
    if sample_noise:
        noise_examples = f"""
The user has confirmed these are examples of things that are NOT part numbers (ignore these and anything similar):
  {sample_noise}"""

    prompt = f"""You are an expert at identifying part numbers in unstructured text from construction and heavy equipment contexts.

Extract ALL individual part numbers from the text below. Part numbers come in many formats:
- Letter-dash-digits: KMP-8347, AT-39571, 1R-0750
- Letter-prefix no dash: T165404, RE505980, AH212096
- Multi-segment dashed: 6754-81-8180, 20Y-32-00300, 707-01-0K620
- Digit-dash-digits: 259-0815, 4333040
- Dotted formats: 3E.1234, 5P.7890
- Slash-separated: 123/4567
- Any other manufacturer part number or catalog number
{parts_examples}
Rules:
- Return ONLY a JSON array of strings, each string being one part number.
- Do NOT include any of the following — these are NOT part numbers:
  * Equipment codes matching "XX - ###" (two letters, space-dash-space, digits) e.g. "CA - 401"
  * Years (e.g. 2018, 2019, 2020, 2021)
  * Equipment make names (Caterpillar, John Deere, Komatsu, Volvo, Case, CAT, etc.)
  * Equipment model names/numbers (D6T, 310SL, PC210LC-11, A40G, CX350D, etc.)
  * Measurements or units (lbs, psi, gal, ft, mm, hours, hrs, mph)
  * Dates in any format (01/15/2023, 2023-01-15, etc.)
  * Quantities (qty 5, x3, etc.)
  * Serial numbers, work order numbers, drawing references, image file names, PO numbers
  * Generic English words, descriptions, or maintenance notes
{noise_examples}
- Preserve the exact formatting of each part number as it appears in the text.
- If no part numbers are found, return an empty array: []

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


def extract_parts_with_regex(text_blob, sample_noise=""):
    """Fallback: use regex heuristics to find part numbers in text."""
    # Build a set of noise examples to exclude
    noise_set = set()
    if sample_noise:
        for item in sample_noise.split(","):
            item = item.strip()
            if item:
                noise_set.add(item.upper())

    # Common part number patterns for construction/heavy equipment
    patterns = [
        # Multi-segment dashed: 6754-81-8180, 20Y-32-00300, 707-01-0K620
        r'\b\d{1,4}[A-Z]?-\d{2,4}-(?:[A-Z0-9]{3,8})\b',
        # Letter-prefix with dash: KMP-8347, AT-39571, 1R-0750
        r'\b[A-Z]{1,5}-\d{3,10}\b',
        # Digit-letter-dash: 1R-0750
        r'\b\d[A-Z]-\d{3,6}\b',
        # Digit-dash-digits (3+ digits each side): 259-0815
        r'\b\d{3,5}-\d{3,8}\b',
        # Letter-prefix no dash (2+ letters then 4+ digits): RE505980, AH212096, KRA1921
        r'\b[A-Z]{2,5}\d{4,10}\b',
        # Single letter prefix (1 letter then 5+ digits): T165404
        r'\b[A-Z]\d{5,10}\b',
        # Mixed alphanumeric segments: AB12C34, 20Y32
        r'\b[A-Z]{1,3}\d{2,4}[A-Z]\d{2,4}\b',
        # Pure digits 6+ long (catches standalone catalog numbers): 4333040
        r'\b\d{6,12}\b',
        # Dotted format: 3E.1234
        r'\b[A-Z0-9]{1,4}\.[A-Z0-9]{3,8}\b',
        # Slash-separated (but not dates like 01/15/2024): 123/4567
        r'\b\d{2,5}/\d{5,8}\b',
    ]

    # Exclusion filters
    # Equipment codes have spaces around dash: "CA - 401", "JD - 102"
    equipment_code_re = re.compile(r'^[A-Za-z]{2}\s+-\s+\d{1,5}$')
    year_re = re.compile(r'^(19|20)\d{2}$')
    # Common equipment model numbers to exclude
    model_names = {
        'D6T', '310SL', 'PC210LC', 'A40G', 'CX350D', 'D6', 'D8', 'D9',
        'D10', 'D11', '320', '330', '345', '349', '390',
    }

    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text_blob):
            val = match.group().strip()
            # Skip equipment codes
            if equipment_code_re.match(val):
                continue
            # Skip years
            if year_re.match(val):
                continue
            # Skip known model numbers
            if val.upper() in model_names:
                continue
            # Skip user-specified noise examples
            if val.upper() in noise_set:
                continue
            found.add(val)

    return sorted(found)


def filter_extracted_parts(parts, equip_name="", equip_code="", sample_noise=""):
    """Post-process extracted parts to remove false positives."""
    # Equipment codes have spaces around dash: "CA - 401", "JD - 102"
    equipment_code_re = re.compile(r'^[A-Za-z]{2}\s+-\s+\d{1,5}$')
    year_re = re.compile(r'^(19|20)\d{2}$')
    pure_word_re = re.compile(r'^[A-Za-z]+$')
    # Common noise words that might slip through
    noise_words = {
        'bom', 'qty', 'pcs', 'n/a', 'na', 'none', 'see', 'notes', 'also',
        'the', 'and', 'for', 'with', 'from', 'last', 'next', 'new', 'old',
        'lbs', 'psi', 'gal', 'hrs', 'mph', 'rpm', 'ft', 'mm', 'in',
    }
    # Noise patterns from user-provided samples (detect prefixes like WO-, DWG-, IMG_, PO#)
    noise_prefixes = set()
    noise_exact = set()
    if sample_noise:
        for item in sample_noise.split(","):
            item = item.strip()
            if not item:
                continue
            noise_exact.add(item.upper())
            # Extract prefix pattern (letters/symbols before first digit or underscore)
            prefix_match = re.match(r'^([A-Za-z]+[-_#])', item)
            if prefix_match:
                noise_prefixes.add(prefix_match.group(1).upper())
    # Extract tokens from equipment name to exclude model numbers
    name_tokens = set()
    if equip_name:
        name_tokens = {t.upper() for t in re.split(r'[\s,]+', equip_name) if t}

    filtered = []
    seen = set()
    for part in parts:
        p = part.strip()
        if not p:
            continue
        upper = p.upper()
        # Dedup
        if upper in seen:
            continue
        seen.add(upper)
        # Skip equipment codes
        if equipment_code_re.match(p):
            continue
        # Skip if it matches the equipment code from this row
        if equip_code and p.replace(" ", "") == equip_code.replace(" ", ""):
            continue
        # Skip years
        if year_re.match(p):
            continue
        # Skip pure English words
        if pure_word_re.match(p) and p.lower() in noise_words:
            continue
        # Skip if it's a token from the equipment name (model number, make)
        if upper in name_tokens:
            continue
        # Skip very short values (1-2 chars are unlikely to be part numbers)
        if len(p) <= 2:
            continue
        # Skip user-specified noise examples (exact match)
        if upper in noise_exact:
            continue
        # Skip values matching noise prefixes (e.g., WO-, DWG-, IMG_, PO#)
        if any(upper.startswith(prefix) for prefix in noise_prefixes):
            continue
        # Skip common non-part artifacts: file names, serial number labels
        if re.search(r'\.(jpg|jpeg|png|pdf|doc|docx|xls|xlsx)$', p, re.I):
            continue
        # Skip date fragments: dd/yyyy, mm/yyyy, yyyy-mm
        if re.match(r'^\d{1,2}/\d{4}$', p) or re.match(r'^\d{4}-\d{2}$', p):
            continue
        # Skip values that are a substring of a noise example (e.g., "2024-0451" from "WO-2024-0451")
        if noise_exact:
            is_noise_fragment = False
            for noise_val in noise_exact:
                if upper in noise_val or noise_val.endswith(upper):
                    is_noise_fragment = True
                    break
            if is_noise_fragment:
                continue
        filtered.append(p)

    return filtered


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
  python extract_parts.py input.csv -o parts_output.csv \\
      --sample-parts "KMP-8347, 259-0815, 6754-81-8180, T165404, VOE11709634" \\
      --sample-noise "WO-2024-0451, DWG-4410, IMG_4521.jpg, PO# 20240087, SN: CAT00D6T"

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
    parser.add_argument(
        "--sample-parts",
        default="",
        help="Comma-separated examples of real part numbers from your data, "
             "so the extractor knows what to look for "
             '(e.g., "KMP-8347, 6754-81-8180, T165404, VOE11709634")',
    )
    parser.add_argument(
        "--sample-noise",
        default="",
        help="Comma-separated examples of things that look like part numbers "
             "but should be IGNORED (work orders, serial numbers, drawing refs, "
             'file names, etc.) e.g., "WO-2024-0451, DWG-4410, IMG_4521.jpg"',
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

    sample_parts = args.sample_parts.strip()
    sample_noise = args.sample_noise.strip()

    use_llm = bool(args.api_key)
    if use_llm:
        print(f"  Using OpenRouter model: {args.model}")
    else:
        print("  No API key — using regex fallback for part extraction.")
        print("  For better results, set OPENROUTER_API_KEY or use --api-key.")

    if sample_parts:
        print(f"  Sample part numbers: {sample_parts}")
    if sample_noise:
        print(f"  Sample noise to ignore: {sample_noise}")

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
            raw_parts = extract_parts_with_llm(
                text_blob, args.api_key, args.model, sample_parts, sample_noise
            )
        else:
            raw_parts = extract_parts_with_regex(text_blob, sample_noise)

        parts = filter_extracted_parts(raw_parts, equip_name, equip_code, sample_noise)
        if len(raw_parts) != len(parts):
            print(f"  (filtered {len(raw_parts) - len(parts)} false positive(s))")
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
