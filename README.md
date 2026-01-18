# fastwebfetch

Fastwebfetch scrapes web pages with Patchright and converts extracted HTML into cleaned Markdown. It relies on the `markdownify`, `mdformat`, `tqdm`, `pylatexenc`, and `patchright` packages.

## How to use (one-shot)

The fastest way to run once and get Markdown output:

```bash
uv sync
uv run main.py init
uv run main.py "https://example.com" > output.md
```

If you already ran `init`, just run:

```bash
uv run main.py "https://example.com" > output.md
```

## MCP quickstart

Run the MCP server directly from the repo URL:

```bash
uvx --from "git+https://github.com/trotsky1997/Fast-Web-Fetch.git" python -m mcp_server
```

Example client config (stdio server):

```json
{
  "servers": {
    "fastwebfetch": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/trotsky1997/Fast-Web-Fetch.git",
        "python",
        "-m",
        "mcp_server"
      ]
    }
  }
}
```

## Features

- Patchright-based scraping with a persistent browser profile.
- Optional paywall bypass via `bypass-paywalls-chrome-clean`.
- Markdown cleanup with optional URL stripping and summary mode.

## Requirements

- Python 3.11+ (see `.python-version`).
- `uv` for dependency management (recommended).

## uv workflow

- Install uv (see https://astral.sh/uv for installers) or use `pip install uv`.
- Run `uv sync` to install all dependencies from `uv.lock`.
- Use `uv run main.py` to execute the script with uv-managed dependencies.
- When dependencies change, run `uv lock` (or `uv lock --upgrade`), then commit the updated `uv.lock` alongside `pyproject.toml`.

## Initialization

- Run `uv run main.py init` as a one-shot setup command to install Chromium via Patchright, create the persistent profile directory, and validate that the bundled `extensions/bypass-paywalls-chrome-clean` assets exist.
- When the bypass extension is missing, init downloads it automatically from GitFlic; override the URL with `FASTWEBFETCH_PAYWALL_URL` if needed.
- The `init` command accepts the same `--browser-data-dir` as the scraper, plus `--channel` (defaults to `chromium`) to choose the Patchright browser and `--paywall-extension-dir` when you moved the bypass extension.
- Skip the automatic browser install with `--skip-chrome-install` (useful when the requested browser is already available). Re-running `init` is safe and simply updates the profile directory and reconfirms the assets.

## Patchright scraping

- Install Chrome through patchright with `patchright install chrome` (recommended) or install Chromium and run `uv run main.py init --channel chromium`.
- Run `uv run main.py <URL>` to scrape a page, convert it to Markdown, and stream the result on stdout; pass `--browser-data-dir` to reuse a persistent profile and `--no-headless` to show the UI.
- The scraper launches Patchright with a persistent context (`no_viewport=True`) and uses the Chrome channel by default.
- Enable paywall bypassing with `--enable-paywall-bypass`, or point to a custom directory via `--paywall-extension-dir`.
- Extensions are disabled in headless Chromium/Chrome, so avoid `--headless` when relying on paywall bypassing.
- See `extensions/bypass-paywalls-chrome-clean/README.md` for the paywall extension configuration, updates, and bundled MIT license.
- URLs are stripped from the markdown output by default to avoid link noise; add `--keep-urls` if you need to preserve HTTP/HTTPS links in the converted content.
- Use `--summary-only` to generate a short model summary instead of full markdown. The default checkpoint is `HuggingFaceTB/SmolLM2-135M-Instruct`, adjustable via `--summary-model`.

## Quickstart

```bash
uv sync
uv run main.py init
uv run main.py "https://example.com"
```

## Development

- Use `uv run pytest` for tests once they exist.
- Format code with `uv run black main.py` and lint with `uv run ruff main.py`.

## MCP usage

### Run via uvx (git path)

Use `uvx` to run the MCP server directly from a git URL without cloning:

```bash
uvx --from "git+https://github.com/trotsky1997/Fast-Web-Fetch.git" python -m mcp_server
```

Notes:
- Update the git URL to your fork or a specific ref if needed.
- `uvx` resolves dependencies from `pyproject.toml` in the repo.
- The server uses a persistent Patchright profile at `~/.fastwebfetch/patchright-profile` by default.

### MCP config (generic JSON, copy/paste)

Use this JSON in any MCP client that supports stdio servers:

```json
{
  "servers": {
    "fastwebfetch": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/trotsky1997/Fast-Web-Fetch.git",
        "python",
        "-m",
        "mcp_server"
      ]
    }
  }
}
```

### fetch_md tool

The MCP server exposes a single tool named `fetch_md` with these inputs:

- `url` (string, required): URL to scrape.
- `enable_paywall_bypass` (boolean, optional): Load the bundled bypass extension.
- `keep_urls` (boolean, optional): Keep HTTP/HTTPS URLs in the markdown output.
- `summary_only` (boolean, optional): Return a short summary instead of full markdown.
