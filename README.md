# Equipment Info Finder

Reads a CSV or Markdown file of construction equipment/vehicles listed by **make, model, and year**, searches the internet for their specifications, and appends the results back to the file.

## Specifications Retrieved

- Weight (lbs)
- Length, Width, Height (inches)
- Number of Axles
- Fuel Type
- Engine Power (HP) & Engine Model
- Operating Capacity
- Max Speed (MPH)
- Fuel Capacity (gallons)
- Additional Specs

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Basic (regex-based parsing, no API key needed)

```bash
python equipment_finder.py sample_equipment.csv
```

### With OpenRouter API (recommended for better accuracy)

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
python equipment_finder.py sample_equipment.csv
```

Or pass the key directly:

```bash
python equipment_finder.py sample_equipment.csv --api-key sk-or-v1-your-key-here
```

### Markdown input/output

```bash
python equipment_finder.py sample_equipment.md
python equipment_finder.py sample_equipment.md -o results.md
```

### Output to a different file

```bash
python equipment_finder.py sample_equipment.csv -o results.csv
```

### Resume a partial run (skip entries that already have specs)

```bash
python equipment_finder.py results.csv --skip-existing
```

### Choose an OpenRouter model

```bash
python equipment_finder.py sample_equipment.csv --model anthropic/claude-sonnet-4
```

## Input File Format

### CSV

```csv
make,model,year
Caterpillar,D6T Dozer,2020
John Deere,310SL Backhoe Loader,2019
```

### Markdown

```markdown
| Make | Model | Year |
|------|-------|------|
| Caterpillar | D6T Dozer | 2020 |
| John Deere | 310SL Backhoe Loader | 2019 |
```

## How It Works

1. Parses the input CSV or Markdown file for equipment entries (preserving any existing spec data)
2. For each entry, runs multiple search queries via DuckDuckGo for broader spec coverage
3. Fetches up to 4 top results per entry with automatic retry on network errors
4. Parses specifications using either:
   - **OpenRouter API** (recommended) — sends the raw text to an LLM for structured extraction
   - **Regex fallback** — pattern-matches common spec formats when no API key is provided
5. Writes the enriched data back to the output file with a coverage summary report

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | overwrites input | Output file path (.csv or .md) |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--model` | `google/gemini-2.0-flash-001` | OpenRouter model for parsing |
| `--delay` | `2.0` | Seconds between lookups |
| `--skip-existing` | off | Skip entries that already have spec data |
