import asyncio
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    print(
        "Error: mcp package is not installed. Please install it with: pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

from main import fetch_page_html, md, run_init, summarize_text


server = Server("fastwebfetch")
_INIT_MARKER_NAME = ".init_complete"
_init_attempted = False
_DEFAULT_USER_DATA_DIR = Path.home() / ".fastwebfetch" / "patchright-profile"


def _ensure_initialized(browser_data_dir: str) -> None:
    global _init_attempted
    if _init_attempted:
        return

    user_data_path = Path(browser_data_dir)
    marker_path = user_data_path / _INIT_MARKER_NAME
    if marker_path.exists():
        _init_attempted = True
        return

    user_data_path.mkdir(parents=True, exist_ok=True)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            run_init(["--browser-data-dir", str(user_data_path)])
        marker_path.write_text("ok", encoding="utf-8")
    finally:
        _init_attempted = True


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_md",
            description="Fetch a URL using Patchright and return cleaned Markdown or a summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape."},
                    "enable_paywall_bypass": {
                        "type": "boolean",
                        "description": "Load the bundled bypass extension before scraping.",
                        "default": False,
                    },
                    "keep_urls": {
                        "type": "boolean",
                        "description": "Keep HTTP/HTTPS URLs in the markdown output.",
                        "default": False,
                    },
                    "summary_only": {
                        "type": "boolean",
                        "description": "Return a short summary instead of full markdown.",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Optional[dict[str, Any]]
) -> list[types.TextContent]:
    if not arguments or "url" not in arguments:
        raise ValueError("Missing required argument: url")

    if name != "fetch_md":
        raise ValueError(f"Unknown tool: {name}")

    url = arguments["url"]
    if not isinstance(url, str):
        raise ValueError("url must be a string")

    enable_paywall_bypass = arguments.get("enable_paywall_bypass", False)
    keep_urls = arguments.get("keep_urls", False)
    summary_only = arguments.get("summary_only", False)

    if not isinstance(enable_paywall_bypass, bool):
        raise ValueError("enable_paywall_bypass must be a boolean")
    if not isinstance(keep_urls, bool):
        raise ValueError("keep_urls must be a boolean")
    if not isinstance(summary_only, bool):
        raise ValueError("summary_only must be a boolean")

    try:
        browser_data_dir = _DEFAULT_USER_DATA_DIR
        _ensure_initialized(str(browser_data_dir))
        html = fetch_page_html(
            url,
            user_data_dir=browser_data_dir,
            channel="chrome",
            headless=True,
            bypass_extension_dir=Path("extensions/bypass-paywalls-chrome-clean")
            if enable_paywall_bypass
            else None,
            wait_until="domcontentloaded",
            timeout_ms=30000,
        )
        markdown = md(html, strip_urls=not keep_urls)

        if not summary_only:
            return [types.TextContent(type="text", text=markdown)]

        summary = summarize_text(
            markdown,
            checkpoint="HuggingFaceTB/SmolLM2-135M-Instruct",
            device="cpu",
            max_new_tokens=160,
            temperature=0.2,
            top_p=0.9,
        )
        return [types.TextContent(type="text", text=summary)]
    except Exception as exc:
        error_msg = f"fastwebfetch MCP error: {exc}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg) from exc


async def main_async() -> None:
    _ensure_initialized(str(_DEFAULT_USER_DATA_DIR))
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fastwebfetch",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
