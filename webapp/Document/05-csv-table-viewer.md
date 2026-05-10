# CSV table viewer (`/csv-table`)

## Purpose

Full-page, scrollable table for inspecting pipeline CSV/TSV outputs (including large detection tables) without downloading manually.

## Files

- **HTML**: `templates/csv_table.html`
- **JS**: `static/csv_table.js`

Query parameters passed from other pages:

- **`path`** — Required server-side path to the file (URL-encoded).
- **`video_path`**, **`project`** — Optional; used only for UI “back” / context labels (see template/JS).

## API

**`GET /api/csv_table`**

| Query param | Default | Behaviour |
|-------------|---------|-----------|
| `path` | — | Required. Resolved with **`_safe_path_any`** (must be under allowed roots). |
| `max_rows` | `25000` | Clamped to **`[1, 100_000]`**. |

Behaviour:

1. Rejects non-files and extensions other than **`.csv`**, **`.tsv`**, **`.txt`**.
2. Reads UTF-8 with **`errors="replace"`**.
3. **`_guess_csv_delimiter`** on first non-empty line: prefers tab if dominant, else `;` vs `,`.
4. First row → **`headers`**; remaining rows → **`rows`** (list of string lists).
5. If row count would exceed **`max_rows`**, response sets **`truncated: true`** and omits the overflow row.

Response JSON includes **`delimiter`**, **`row_count_returned`**, **`max_rows`**, and resolved **`path`**.

## Security note

Same as other file APIs: arbitrary paths outside **`ALLOWED_PATH_ROOTS`** return **400** from **`_safe_path_any`**.
