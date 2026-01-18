# fastwebfetch

This project converts extracted HTML into cleaned Markdown. It relies on the `datasets`, `markdownify`, `mdformat`, `tqdm`, `pylatexenc`, and `patchright` packages.

## uv workflow

- Install uv (see https://astral.sh/uv for installers) or use `pip install uv`.
- Run `uv sync` to install all dependencies from `uv.lock`.
- Use `uv run main.py` to execute the script with uv-managed dependencies.
- When dependencies change, run `uv lock` (or `uv lock --upgrade`), then commit the updated `uv.lock` alongside `pyproject.toml`.

## Initialization

-- Run `uv run main.py init` as a one-shot setup command to install Chromium via Patchright, create the persistent profile directory, and validate that the bundled `extensions/bypass-paywalls-chrome-clean` assets exist.
- The init command now downloads the GitFlic ZIP (`bypass-paywalls-chrome-clean-master.zip`) into `extensions/bypass-paywalls-chrome-clean` when that directory is absent; re-running init refreshes the assets. Use `--paywall-release-url` to pin a specific ZIP release, or `--skip-extension-check` when you prefer to manage the extension manually.
- For manual installs, download the same GitFlic ZIP, unzip it into your desired location (keep the directory name stable), and either point to it with `--paywall-extension-dir` or skip the automated check.
- The `init` command accepts the same `--browser-data-dir` as the scraper, plus `--channel` (defaults to `chromium`) to choose the Patchright browser and `--paywall-extension-dir` when you moved the bypass extension.
- Skip the automatic browser install with `--skip-chrome-install` (useful when the requested browser is already available). Re-running `init` is safe and simply updates the profile directory and reconfirms the assets.

## Patchright scraping

- Install Chromium through patchright (`patchright install chromium`) or any other preferred installer and keep the channel set to `chromium` so the browser can be launched without fingerprinting shields (use `--channel chrome` if you need the real Google Chrome binary instead).
- Run `uv run main.py <URL>` to scrape a page, convert it to Markdown, and stream the result on stdout; pass `--browser-data-dir` to reuse a persistent profile and `--headless` to drop the UI when you don’t need it.
- The script uses `patchright.chromium.launch_persistent_context(..., channel="chromium", no_viewport=True)` as the recommended undetectable configuration. Avoid adding custom headers, and let Chromium handle the default fingerprinting surface.
- The scraper loads `extensions/bypass-paywalls-chrome-clean` by default so you can surf paywalled sites. Skip it with `--disable-paywall-bypass`, or point to a custom directory via `--paywall-extension-dir`.
- Extensions are disabled in headless Chromium/Chrome, so avoid `--headless` when relying on paywall bypassing.
- See `extensions/bypass-paywalls-chrome-clean/README.md` for the paywall extension configuration, updates, and bundled MIT license.
- URLs are stripped from the markdown output by default to avoid link noise; add `--keep-urls` if you need to preserve HTTP/HTTPS links in the converted content.
- Use `--summary-only` to generate a short model summary instead of full markdown. The default checkpoint is `HuggingFaceTB/SmolLM2-135M-Instruct`, adjustable via `--summary-model`.

## Development

- Use `uv run pytest` for tests once they exist.
- Format code with `uv run black main.py` and lint with `uv run ruff main.py`.
