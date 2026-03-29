#!/usr/bin/env python3
"""
Equipment Specification Finder

Reads a CSV or Markdown file containing construction equipment/vehicles
(make, model, year), searches the internet for specifications, and appends
the found specs back to the file.

Uses OpenRouter API for intelligent parsing of search results into
structured specification data.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

# Specification fields we attempt to populate
SPEC_FIELDS = [
    "weight_lbs",
    "length_in",
    "width_in",
    "height_in",
    "number_of_axles",
    "fuel_type",
    "engine_power_hp",
    "engine_model",
    "operating_capacity",
    "max_speed_mph",
    "fuel_capacity_gal",
    "additional_specs",
]

SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_csv(filepath):
    """Read equipment entries from a CSV file."""
    entries = []
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "make": row.get("make", "").strip(),
                "model": row.get("model", "").strip(),
                "year": row.get("year", "").strip(),
            })
    return entries


def read_markdown(filepath):
    """Read equipment entries from a Markdown table file."""
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    header_indices = {}
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        if not in_table:
            # First row with pipes is the header
            for i, cell in enumerate(cells):
                lower = cell.lower()
                if "make" in lower:
                    header_indices["make"] = i
                elif "model" in lower:
                    header_indices["model"] = i
                elif "year" in lower:
                    header_indices["year"] = i
            in_table = True
            continue

        # Skip separator rows (e.g., |---|---|---|)
        if all(re.match(r"^[-: ]+$", c) for c in cells):
            continue

        if header_indices:
            entries.append({
                "make": cells[header_indices.get("make", 0)].strip(),
                "model": cells[header_indices.get("model", 1)].strip(),
                "year": cells[header_indices.get("year", 2)].strip(),
            })

    return entries


def write_csv(filepath, entries):
    """Write equipment entries with specs to a CSV file."""
    fieldnames = ["make", "model", "year"] + SPEC_FIELDS
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {k: entry.get(k, "") for k in fieldnames}
            writer.writerow(row)


def write_markdown(filepath, entries):
    """Write equipment entries with specs to a Markdown table file."""
    fieldnames = ["make", "model", "year"] + SPEC_FIELDS
    headers = [f.replace("_", " ").title() for f in fieldnames]

    lines = ["# Equipment Specifications\n\n"]
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")

    for entry in entries:
        cells = [str(entry.get(f, "")) for f in fieldnames]
        lines.append("| " + " | ".join(cells) + " |\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(query, num_results=5):
    """Search using DuckDuckGo HTML and return a list of (title, url, snippet) tuples."""
    results = []
    try:
        url = "https://html.duckduckgo.com/html/"
        resp = requests.post(
            url,
            data={"q": query},
            headers=SEARCH_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for result_div in soup.select(".result")[:num_results]:
            title_tag = result_div.select_one(".result__title a")
            snippet_tag = result_div.select_one(".result__snippet")
            if title_tag:
                title = title_tag.get_text(strip=True)
                href = title_tag.get("href", "")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append((title, href, snippet))
    except Exception as e:
        print(f"  [!] Search error: {e}", file=sys.stderr)

    return results


def fetch_page_text(url, max_chars=12000):
    """Fetch a web page and return its visible text, truncated."""
    try:
        resp = requests.get(url, headers=SEARCH_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception as e:
        print(f"  [!] Fetch error for {url}: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# OpenRouter LLM integration
# ---------------------------------------------------------------------------

def parse_specs_with_llm(equipment_description, raw_text, api_key, model):
    """Use OpenRouter API to extract structured specs from raw web text."""
    prompt = f"""You are a construction equipment specification expert.

Given the following equipment: {equipment_description}

Extract specifications from the text below. Return ONLY a valid JSON object with these fields (use empty string "" if not found):
- weight_lbs: operating weight in pounds (convert from kg if needed: 1 kg = 2.205 lbs)
- length_in: overall length in inches (convert from mm if needed: 1 mm = 0.03937 in)
- width_in: overall width in inches (convert from mm if needed)
- height_in: overall height in inches (convert from mm if needed)
- number_of_axles: number of axles (integer as string)
- fuel_type: fuel type (e.g., Diesel, Gasoline, Electric, Hybrid)
- engine_power_hp: engine horsepower (convert from kW if needed: 1 kW = 1.341 hp)
- engine_model: engine make and model
- operating_capacity: rated/operating capacity with units
- max_speed_mph: maximum travel speed in mph
- fuel_capacity_gal: fuel tank capacity in gallons (convert from liters if needed: 1 L = 0.2642 gal)
- additional_specs: any other notable specs as a brief comma-separated string

Raw text from specification sources:
---
{raw_text[:8000]}
---

Return ONLY the JSON object, no markdown fences, no explanation."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 800,
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

        specs = json.loads(content)
        return specs
    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse error from LLM response: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"  [!] OpenRouter API error: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Fallback regex-based parser (no API key needed)
# ---------------------------------------------------------------------------

def parse_specs_with_regex(text):
    """Best-effort regex extraction of specs from raw text."""
    specs = {f: "" for f in SPEC_FIELDS}

    # Weight patterns
    w = re.search(r"(?:operat|gross|total|service)\w*\s*weight[:\s]*[~≈]*([\d,]+)\s*(?:lbs?|pounds)", text, re.I)
    if w:
        specs["weight_lbs"] = w.group(1).replace(",", "")
    else:
        w = re.search(r"(?:operat|gross|total|service)\w*\s*weight[:\s]*[~≈]*([\d,]+)\s*kg", text, re.I)
        if w:
            specs["weight_lbs"] = str(int(float(w.group(1).replace(",", "")) * 2.205))

    # Length
    m = re.search(r"(?:overall\s+)?length[:\s]*([\d,.]+)\s*(?:in|inches)", text, re.I)
    if m:
        specs["length_in"] = m.group(1)
    else:
        m = re.search(r"(?:overall\s+)?length[:\s]*([\d,.]+)\s*mm", text, re.I)
        if m:
            specs["length_in"] = str(round(float(m.group(1).replace(",", "")) * 0.03937, 1))

    # Width
    m = re.search(r"(?:overall\s+)?width[:\s]*([\d,.]+)\s*(?:in|inches)", text, re.I)
    if m:
        specs["width_in"] = m.group(1)
    else:
        m = re.search(r"(?:overall\s+)?width[:\s]*([\d,.]+)\s*mm", text, re.I)
        if m:
            specs["width_in"] = str(round(float(m.group(1).replace(",", "")) * 0.03937, 1))

    # Height
    m = re.search(r"(?:overall\s+)?height[:\s]*([\d,.]+)\s*(?:in|inches)", text, re.I)
    if m:
        specs["height_in"] = m.group(1)
    else:
        m = re.search(r"(?:overall\s+)?height[:\s]*([\d,.]+)\s*mm", text, re.I)
        if m:
            specs["height_in"] = str(round(float(m.group(1).replace(",", "")) * 0.03937, 1))

    # Axles
    m = re.search(r"(\d)\s*axles?", text, re.I)
    if m:
        specs["number_of_axles"] = m.group(1)

    # Fuel type
    m = re.search(r"fuel\s*type[:\s]*(diesel|gasoline|electric|hybrid|gas|biodiesel)", text, re.I)
    if m:
        specs["fuel_type"] = m.group(1).capitalize()
    elif re.search(r"diesel\s*engine", text, re.I):
        specs["fuel_type"] = "Diesel"

    # Engine power
    m = re.search(r"([\d,.]+)\s*(?:gross\s+)?(?:hp|horsepower)", text, re.I)
    if m:
        specs["engine_power_hp"] = m.group(1)
    else:
        m = re.search(r"([\d,.]+)\s*kW", text)
        if m:
            specs["engine_power_hp"] = str(round(float(m.group(1).replace(",", "")) * 1.341))

    # Engine model
    m = re.search(r"engine\s*(?:model)?[:\s]*((?:Cat|Cummins|Deere|Isuzu|Volvo|Deutz|Yanmar|Kubota|Tier)\s*[\w\s.-]{3,30})", text, re.I)
    if m:
        specs["engine_model"] = m.group(1).strip()

    # Max speed
    m = re.search(r"(?:max|top|travel)\s*speed[:\s]*([\d.]+)\s*mph", text, re.I)
    if m:
        specs["max_speed_mph"] = m.group(1)

    # Fuel capacity
    m = re.search(r"fuel\s*(?:tank\s*)?capacity[:\s]*([\d.]+)\s*gal", text, re.I)
    if m:
        specs["fuel_capacity_gal"] = m.group(1)
    else:
        m = re.search(r"fuel\s*(?:tank\s*)?capacity[:\s]*([\d.]+)\s*(?:L|liters?|litres?)", text, re.I)
        if m:
            specs["fuel_capacity_gal"] = str(round(float(m.group(1)) * 0.2642, 1))

    return specs


# ---------------------------------------------------------------------------
# Main lookup logic
# ---------------------------------------------------------------------------

def lookup_equipment_specs(entry, api_key=None, model=None):
    """Search for specs for a single equipment entry and return enriched entry."""
    make = entry["make"]
    model_name = entry["model"]
    year = entry["year"]
    desc = f"{year} {make} {model_name}"

    print(f"\n{'='*60}")
    print(f"  Looking up: {desc}")
    print(f"{'='*60}")

    # Search for specification pages
    query = f"{desc} specifications weight dimensions"
    results = web_search(query, num_results=5)

    if not results:
        print("  [!] No search results found.")
        return {**entry, **{f: "" for f in SPEC_FIELDS}}

    # Fetch text from top results
    combined_text = ""
    for i, (title, url, snippet) in enumerate(results[:3]):
        print(f"  [{i+1}] {title}")
        page_text = fetch_page_text(url, max_chars=6000)
        if page_text:
            combined_text += f"\n--- Source: {title} ---\n{page_text}\n"
        time.sleep(1)  # polite delay

    if not combined_text:
        print("  [!] Could not fetch any page content.")
        return {**entry, **{f: "" for f in SPEC_FIELDS}}

    # Parse specs
    if api_key:
        print("  Parsing specs with LLM via OpenRouter...")
        specs = parse_specs_with_llm(desc, combined_text, api_key, model)
        if not specs:
            print("  [!] LLM parsing failed, falling back to regex.")
            specs = parse_specs_with_regex(combined_text)
    else:
        print("  Parsing specs with regex (no API key provided)...")
        specs = parse_specs_with_regex(combined_text)

    # Merge
    enriched = {**entry}
    for field in SPEC_FIELDS:
        enriched[field] = specs.get(field, "")

    # Show summary
    found = sum(1 for f in SPEC_FIELDS if enriched.get(f))
    print(f"  => Found {found}/{len(SPEC_FIELDS)} spec fields.")
    return enriched


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Equipment Specification Finder — looks up specs for construction equipment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python equipment_finder.py sample_equipment.csv
  python equipment_finder.py sample_equipment.md -o results.csv
  python equipment_finder.py equipment.csv --api-key sk-or-... --model google/gemini-2.0-flash-001

Environment variables:
  OPENROUTER_API_KEY   API key for OpenRouter (alternative to --api-key)
""",
    )

    parser.add_argument(
        "input_file",
        help="Path to input CSV or Markdown file containing equipment list",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: overwrites input file). "
             "Format is inferred from extension (.csv or .md).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model to use for parsing (default: google/gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between equipment lookups (default: 2.0)",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output or input_file

    # Detect format
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".csv":
        entries = read_csv(input_file)
    elif ext in (".md", ".markdown"):
        entries = read_markdown(input_file)
    else:
        print(f"Error: Unsupported file format '{ext}'. Use .csv or .md", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("No equipment entries found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(entries)} equipment entries from {input_file}")

    if args.api_key:
        print(f"Using OpenRouter model: {args.model}")
    else:
        print("No OpenRouter API key provided — using regex-based parsing.")
        print("For better results, set OPENROUTER_API_KEY or use --api-key.")

    # Process each entry
    enriched_entries = []
    for i, entry in enumerate(entries):
        enriched = lookup_equipment_specs(entry, args.api_key, args.model)
        enriched_entries.append(enriched)
        if i < len(entries) - 1:
            time.sleep(args.delay)

    # Write output
    out_ext = os.path.splitext(output_file)[1].lower()
    if out_ext == ".csv":
        write_csv(output_file, enriched_entries)
    elif out_ext in (".md", ".markdown"):
        write_markdown(output_file, enriched_entries)
    else:
        write_csv(output_file, enriched_entries)

    print(f"\n{'='*60}")
    print(f"  Results written to: {output_file}")
    print(f"  Total entries processed: {len(enriched_entries)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
