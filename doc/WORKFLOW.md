# fastwebfetch Workflow

This document explains the main flow and module responsibilities in fastwebfetch, so you can understand how the script and MCP service work.

## High-level flow

1. Initialization: install Patchright browsers and prepare the persistent user data directory.
2. Fetching: launch a persistent Chromium/Chrome context with Patchright, visit the URL, and capture HTML.
3. Conversion: turn HTML into Markdown and clean it up.
4. Output: print Markdown in CLI mode; return Markdown or a summary in MCP mode.

## Initialization (main.py init)

- `run_init` creates the browser data directory and calls `patchright install <channel>`.
- Use `--browser-data-dir` to set the persistent directory (default: `.patchright-profile`).
- Use `--channel` to choose the Patchright browser channel (default: `chromium`).
- If the browser is already installed, use `--skip-chrome-install` to skip installation.
- Use `--paywall-extension-dir` to set the extension path, and `--skip-extension-check` to skip validation.

## Fetching flow (main.py)

- `fetch_page_html` uses `patchright.sync_api`:
  - `launch_persistent_context` keeps login state and cookies.
  - `headless` controls whether a UI window is shown.
  - When extensions are enabled, it loads them via `--disable-extensions-except` / `--load-extension`.
- `fetch_page_html_async` is the async variant used by the MCP server.

## Markdown conversion

The `md` function applies a fixed sequence of transformations:

1. `markdownify` converts HTML to base Markdown.
2. `LatexNodes2Text(math_mode="with-delimiters")` converts LaTeX expressions into readable text while preserving `$...$` / `$$...$$`.
3. `replace_image_tags_with_alt_text` replaces `![alt](url)` with the `alt` text to avoid image noise.
4. If `strip_urls=True`, `replace_link_tags_with_alt_text` turns `[text](url)` into `text`, removing URLs.
5. `mdformat.text` normalizes Markdown formatting and whitespace.
6. A batch of HTML entities (for example `&nbsp;`, `&quot;`, `&alpha;`) are replaced with readable characters, including math symbols and Greek letters.
7. The placeholder `enter image description here` is replaced with `No image description available` to avoid empty alt text.

Parameter notes:

- `strip_urls`: controls whether link URLs are removed from Markdown (default True). When False, link URLs are preserved.

## Summary mode

- `--summary-only` enables summary output.
- `summarize_text` uses a Transformers model to produce 3-5 concise bullet points.
- The default model is `HuggingFaceTB/SmolLM2-135M-Instruct`, overridable via flags.

## CLI vs MCP behavior

- CLI (`main.py`)
  - Uses the Chrome channel by default (`channel="chrome"`).
  - Only loads the paywall bypass extension when `--enable-paywall-bypass` is set.
  - Prints output to stdout.

- MCP (`mcp_server.py`)
  - Ensures initialization on startup (writes a `.init_complete` marker).
  - Uses a default data directory at `~/.fastwebfetch/patchright-profile`.
  - Runs headless by default for service usage.
  - Exposes the `fetch_md` tool to return Markdown or a summary.

## Key modules

- `main.py`
  - Fetching and Markdown conversion logic.
  - CLI argument definitions and handling.
- `mcp_server.py`
  - MCP server setup and the `fetch_md` tool.
  - MCP-specific initialization and async fetching.
