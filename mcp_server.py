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
            description="Fetch a URL using Patchright and return cleaned Markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape."},
                    "browser_data_dir": {
                        "type": "string",
                        "description": "Persistent browser profile directory.",
                        "default": ".patchright-profile",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Run the browser without a visible window.",
                        "default": True,
                    },
                    "enable_paywall_bypass": {
                        "type": "boolean",
                        "description": "Load the bundled bypass extension before scraping.",
                        "default": False,
                    },
                    "paywall_extension_dir": {
                        "type": "string",
                        "description": "Path to the paywall bypass extension directory.",
                        "default": "extensions/bypass-paywalls-chrome-clean",
                    },
                    "keep_urls": {
                        "type": "boolean",
                        "description": "Keep HTTP/HTTPS URLs in the markdown output.",
                        "default": False,
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "Playwright wait condition.",
                        "default": "domcontentloaded",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Navigation timeout in milliseconds.",
                        "default": 30000,
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="summary_only",
            description="Fetch a URL and return a short model-generated summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape."},
                    "browser_data_dir": {
                        "type": "string",
                        "description": "Persistent browser profile directory.",
                        "default": ".patchright-profile",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Run the browser without a visible window.",
                        "default": True,
                    },
                    "enable_paywall_bypass": {
                        "type": "boolean",
                        "description": "Load the bundled bypass extension before scraping.",
                        "default": False,
                    },
                    "paywall_extension_dir": {
                        "type": "string",
                        "description": "Path to the paywall bypass extension directory.",
                        "default": "extensions/bypass-paywalls-chrome-clean",
                    },
                    "keep_urls": {
                        "type": "boolean",
                        "description": "Keep HTTP/HTTPS URLs in the markdown output.",
                        "default": False,
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "Playwright wait condition.",
                        "default": "domcontentloaded",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Navigation timeout in milliseconds.",
                        "default": 30000,
                    },
                    "summary_model": {
                        "type": "string",
                        "description": "Model checkpoint to use for summarization.",
                        "default": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    },
                    "summary_device": {
                        "type": "string",
                        "description": "Device for the summarization model (cpu or cuda).",
                        "default": "cpu",
                    },
                    "summary_max_new_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate for the summary.",
                        "default": 160,
                    },
                    "summary_temperature": {
                        "type": "number",
                        "description": "Sampling temperature for the summary generation.",
                        "default": 0.2,
                    },
                    "summary_top_p": {
                        "type": "number",
                        "description": "Top-p nucleus sampling for the summary generation.",
                        "default": 0.9,
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

    if name not in {"fetch_md", "summary_only"}:
        raise ValueError(f"Unknown tool: {name}")

    url = arguments["url"]
    if not isinstance(url, str):
        raise ValueError("url must be a string")

    browser_data_dir = arguments.get("browser_data_dir", ".patchright-profile")
    if not isinstance(browser_data_dir, str):
        raise ValueError("browser_data_dir must be a string")

    headless = arguments.get("headless", True)
    enable_paywall_bypass = arguments.get("enable_paywall_bypass", False)
    paywall_extension_dir = arguments.get(
        "paywall_extension_dir", "extensions/bypass-paywalls-chrome-clean"
    )
    keep_urls = arguments.get("keep_urls", False)
    wait_until = arguments.get("wait_until", "domcontentloaded")
    timeout_ms = arguments.get("timeout_ms", 30000)

    if not isinstance(headless, bool):
        raise ValueError("headless must be a boolean")
    if not isinstance(enable_paywall_bypass, bool):
        raise ValueError("enable_paywall_bypass must be a boolean")
    if not isinstance(paywall_extension_dir, str):
        raise ValueError("paywall_extension_dir must be a string")
    if not isinstance(keep_urls, bool):
        raise ValueError("keep_urls must be a boolean")
    if wait_until not in {"domcontentloaded", "load", "networkidle"}:
        raise ValueError("wait_until must be one of: domcontentloaded, load, networkidle")
    if not isinstance(timeout_ms, int):
        raise ValueError("timeout_ms must be an integer")

    try:
        _ensure_initialized(browser_data_dir)
        html = fetch_page_html(
            url,
            user_data_dir=browser_data_dir,
            channel="chrome",
            headless=headless,
            bypass_extension_dir=Path(paywall_extension_dir)
            if enable_paywall_bypass
            else None,
            wait_until=wait_until,
            timeout_ms=timeout_ms,
        )
        markdown = md(html, strip_urls=not keep_urls)

        if name == "fetch_md":
            return [types.TextContent(type="text", text=markdown)]

        summary = summarize_text(
            markdown,
            checkpoint=arguments.get(
                "summary_model", "HuggingFaceTB/SmolLM2-135M-Instruct"
            ),
            device=arguments.get("summary_device", "cpu"),
            max_new_tokens=arguments.get("summary_max_new_tokens", 160),
            temperature=arguments.get("summary_temperature", 0.2),
            top_p=arguments.get("summary_top_p", 0.9),
        )
        return [types.TextContent(type="text", text=summary)]
    except Exception as exc:
        error_msg = f"fastwebfetch MCP error: {exc}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg) from exc


async def main_async() -> None:
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
