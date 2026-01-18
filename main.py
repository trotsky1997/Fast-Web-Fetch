# -=encoding=utf-8=-
import argparse
import subprocess
import sys
from pathlib import Path

from markdownify import markdownify
import mdformat

import re
from pylatexenc.latex2text import LatexNodes2Text

from patchright.sync_api import sync_playwright
from patchright.async_api import async_playwright


_RT_LOCK_MISSING = object()


def _ensure_resource_tracker_lock() -> None:
    """Patch multiprocess.resource_tracker so its lock exposes _recursion_count."""
    try:
        import multiprocess.resource_tracker as resource_tracker
    except ImportError:
        return

    lock = getattr(resource_tracker._resource_tracker, "_lock", None)
    if lock is None or getattr(lock, "_recursion_count", _RT_LOCK_MISSING) is not _RT_LOCK_MISSING:
        return

    class _LockWithCount:
        __slots__ = ("_lock", "_count")

        def __init__(self, base_lock):
            self._lock = base_lock
            self._count = 0

        def acquire(self, *args, **kwargs):
            result = self._lock.acquire(*args, **kwargs)
            if result:
                self._count += 1
            return result

        def release(self, *args, **kwargs):
            self._lock.release(*args, **kwargs)
            self._count -= 1

        def _recursion_count(self):
            return self._count

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.release()

        def __getattr__(self, name):
            return getattr(self._lock, name)

    resource_tracker._resource_tracker._lock = _LockWithCount(lock)


def _patch_resource_tracker_stop() -> None:
    """Suppress AttributeErrors triggered by missing _recursion_count during shutdown."""
    try:
        import multiprocess.resource_tracker as resource_tracker
    except ImportError:
        return

    if getattr(resource_tracker.ResourceTracker._stop, "__patched__", False):
        return

    original_stop = resource_tracker.ResourceTracker._stop

    def _safe_stop(self, *args, **kwargs):
        try:
            return original_stop(self, *args, **kwargs)
        except AttributeError as exc:
            if "_recursion_count" in str(exc):
                return
            raise

    _safe_stop.__patched__ = True
    resource_tracker.ResourceTracker._stop = _safe_stop


_ensure_resource_tracker_lock()
_patch_resource_tracker_stop()

def replace_image_tags_with_alt_text(md_text):
    # Regex for image tags
    img_pattern = r'!\[([^\]]*)\]\(([^\)]*)\)'
    
    # Replace with alt_text
    new_md_text = re.sub(img_pattern, r'\1', md_text)
    
    return new_md_text

def replace_link_tags_with_alt_text(md_text):
    # Regex for link tags
    link_pattern = r'\[([^\]]*)\]\(([^\)]*)\)'
    
    # Replace with alt_text
    new_md_text = re.sub(link_pattern, r'\1', md_text)
    
    return new_md_text


def replace_math_codes_with_text(md_text):
    # Regex for multiline math tags
    multiline_math_pattern = r'\$\$(.*?)\$\$'
    
    # Replace with plain-text LaTeX output
    new_md_text = re.sub(multiline_math_pattern, lambda m: "\n$$\n"+LatexNodes2Text().latex_to_text(m.group(1)) +"\n$$\n", md_text, flags=re.DOTALL)
    
    return new_md_text

def replace_multiline_math_tags_with_text(md_text):
    # Regex for multiline math tags
    multiline_math_pattern = r'\$\$(.*?)\$\$'
    
    # Replace with plain-text LaTeX output
    new_md_text = re.sub(multiline_math_pattern, lambda m: "\n$$\n"+LatexNodes2Text(math_mode='text').latex_to_text(m.group(1)) +"\n$$\n", md_text, flags=re.DOTALL)
    
    return new_md_text



def replace_math_tags_with_text(md_text):
    # Regex for math tags
    math_pattern = r'\$(.*?)\$'
    
    # Replace with plain-text LaTeX output
    new_md_text = re.sub(math_pattern, lambda m: "$"+LatexNodes2Text(math_mode='text').latex_to_text(m.group(1)) +"$", md_text)
    
    return new_md_text

def remove_http_links(text):
    # Regex for HTTP links
    http_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace all HTTP links with an empty string
    cleaned_text = re.sub(http_pattern, '', text)
    
    return cleaned_text

# def clean(match):
#     # Add your cleanup logic here
#     smiles_string = match.group(0)
#     if '.' in smiles_string:
#         return smiles_string
#     else:
#         cleaned_smiles = f'\n```smiles\n{smiles_string}\n```\n'
#         return cleaned_smiles

# def replace_smiles_with_cleaned(text):
#     # Regex for SMILES strings
#     smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=%#:.]*'

#     # Replace all SMILES strings using re.sub()
#     cleaned_text = re.sub(smiles_pattern, clean, text)

#     return cleaned_text

def md(txt, *, strip_urls: bool = True):

    txt = markdownify(txt)
    
    txt = LatexNodes2Text(math_mode='with-delimiters').latex_to_text(txt)
    
    txt = replace_image_tags_with_alt_text(txt)
    if strip_urls:
        txt = replace_link_tags_with_alt_text(txt)

    txt = mdformat.text(txt)
    txt = txt.replace('lt;','<').replace('gt;','>').replace('amp;','&').replace('quot;','"').replace('apos;',"'").replace('nbsp;',' ').replace('ldquo;','“').replace('rdquo;','”').replace('lsquo;','‘').replace('rsquo;','’').replace('mdash;','—').replace('ndash;','–').replace('times;','×').replace('divide;','÷').replace('leq;','≤').replace('geq;','≥').replace('neq;','≠').replace('infty;','∞').replace('alpha;','α').replace('beta;','β').replace('gamma;','γ').replace('delta;','δ').replace('epsilon;','ε').replace('zeta;','ζ').replace('eta;','η').replace('theta;','θ').replace('iota;','ι').replace('kappa;','κ').replace('lambda;','λ').replace('mu;','μ').replace('nu;','ν').replace('xi;','ξ').replace('omicron;','ο').replace('pi;','π').replace('rho;','ρ').replace('sigma;','σ').replace('tau;','τ').replace('upsilon;','υ').replace('phi;','φ').replace('chi;','χ').replace('psi;','ψ').replace('omega;','ω').replace('Alpha;','Α').replace('Beta;','Β').replace('Gamma;','Γ').replace('Delta;','Δ').replace('Epsilon;','Ε').replace('Zeta;','Ζ').replace('Eta;','Η').replace('Theta;','Θ').replace('Iota;','Ι').replace('Kappa;','Κ').replace('Lambda;','Λ').replace('Mu;','Μ').replace('Nu;','Ν').replace('Xi;','Ξ').replace('Omicron;','Ο').replace('Pi;','Π').replace('Rho;','Ρ').replace('Sigma;','Σ').replace('Tau;','Τ').replace('Upsilon;','Υ').replace('Phi;','Φ').replace('Chi;','Χ').replace('Psi;','Ψ').replace('Omega;','Ω').replace('forall;','∀').replace('part;','∂').replace('exist;','∃').replace('empty;','∅').replace('nabla;','∇').replace('isin;','∈').replace('notin;','∉').replace('ni;','∋').replace('prod;','∏').replace('sum;','∑').replace('minus;','−').replace('lowast;','∗').replace('radic;','√').replace('prop;','∝').replace('infin;','∞').replace('ang;','∠').replace('and;','∧').replace('or;','∨').replace('cap;','∩').replace('cup;','∪').replace('int;','∫').replace('there4;','∴').replace('sim;','∼').replace('cong;','≅').replace('asymp;','≈').replace('ne;','≠').replace('equiv;','≡').replace('le;','≤').replace('ge;','≥').replace('sub;','⊂').replace('sup;','⊃').replace('nsub;','⊄').replace('sube;','⊆').replace('supe;','⊇').replace('oplus;','⊕').replace('otimes;','⊗').replace('perp;','⊥').replace('sdot;','⋅').replace('lceil;','⌈').replace('rceil;','⌉').replace('lfloor;','⌊').replace('rfloor;','⌋').replace('lang;','⟨').replace('rang;','⟩').replace('loz;','◊').replace('spades;','♠').replace('clubs;','♣').replace('hearts;','♥').replace('diams;','♦').replace('quot;','"').replace('amp;','&').replace('lt;','<').replace('gt;','>').replace('nbsp;',' ').replace('iexcl;','¡').replace('cent;','¢').replace('pound;','£').replace('curren;','¤')
    txt = txt.replace('yen;','¥').replace('brvbar;','¦').replace('sect;','§').replace('uml;','¨')
    txt = txt.replace('enter image description here','No image description available')
    return txt


def fetch_page_html(
    url: str,
    user_data_dir: Path | str | None = None,
    *,
    channel: str = "chromium",
    headless: bool = False,
    bypass_extension_dir: Path | str | None = None,
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 30000,
) -> str:
    """Scrape `url` using Patchright's Chromium channel + persistent context."""
    user_data_path = Path(user_data_dir) if user_data_dir else Path(".patchright-profile")
    user_data_path.mkdir(parents=True, exist_ok=True)

    args: list[str] = []
    if bypass_extension_dir:
        extension_path = Path(bypass_extension_dir).expanduser().resolve()
        if not extension_path.exists():
            raise FileNotFoundError(
                f"Bypass extension path does not exist: {extension_path}"
            )

        args.extend(
            [
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
            ]
        )

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_path),
            channel=channel,
            headless=headless,
            no_viewport=True,
            **({"args": args} if args else {}),
        )
        try:
            page = context.new_page()
            try:
                page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return page.content()
        finally:
            context.close()


async def fetch_page_html_async(
    url: str,
    user_data_dir: Path | str | None = None,
    *,
    channel: str = "chromium",
    headless: bool = False,
    bypass_extension_dir: Path | str | None = None,
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 30000,
) -> str:
    """Async variant for event-loop contexts (e.g., MCP server)."""
    user_data_path = Path(user_data_dir) if user_data_dir else Path(".patchright-profile")
    user_data_path.mkdir(parents=True, exist_ok=True)

    args: list[str] = []
    if bypass_extension_dir:
        extension_path = Path(bypass_extension_dir).expanduser().resolve()
        if not extension_path.exists():
            raise FileNotFoundError(
                f"Bypass extension path does not exist: {extension_path}"
            )

        args.extend(
            [
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
            ]
        )

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_path),
            channel=channel,
            headless=headless,
            no_viewport=True,
            **({"args": args} if args else {}),
        )
        try:
            page = await context.new_page()
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            except Exception:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return await page.content()
        finally:
            await context.close()


def run_patchright_install(channel: str = "chromium") -> None:
    """Ensure Patchright's requested browser channel is installed."""
    try:
        subprocess.run(["patchright", "install", channel], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Patchright CLI not found in PATH. Run this command through `uv run` or install "
            "`patchright` into your active interpreter so that `patchright install <channel>` "
            "can execute."
        ) from exc


def run_init(argv: list[str] | None = None) -> None:
    """Convenience command that prepares Patchright and bypass assets."""
    parser = argparse.ArgumentParser(
        prog="main.py init",
        description="Install the requested Patchright browser channel and validate the paywall extension.",
    )
    parser.add_argument(
        "--browser-data-dir",
        default=".patchright-profile",
        help="Persistent user data directory that `main.py` will reuse.",
    )
    parser.add_argument(
        "--channel",
        default="chromium",
        help="Patchright browser channel to install (default: chromium).",
    )
    parser.add_argument(
        "--paywall-extension-dir",
        default="extensions/bypass-paywalls-chrome-clean",
        help="Location of the bundled extension (validated unless --skip-extension-check is set).",
    )
    parser.add_argument(
        "--skip-chrome-install",
        action="store_true",
        help="Skip `patchright install <channel>` (useful when that browser is already installed).",
    )
    parser.add_argument(
        "--skip-extension-check",
        action="store_true",
        help="Skip validating that the bundled paywall bypass extension exists.",
    )
    args = parser.parse_args(argv)

    user_data_path = Path(args.browser_data_dir)
    user_data_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_chrome_install:
        run_patchright_install(args.channel)

    if not args.skip_extension_check:
        extension_path = Path(args.paywall_extension_dir)
        if not extension_path.exists():
            raise FileNotFoundError(
                f"Expected bypass extension assets at {extension_path}, but the directory is missing."
            )

    print("Initialization finished.")
    print(f"  · Browser data dir ready at {user_data_path.resolve()}")
    if not args.skip_extension_check:
        print(f"  · Paywall extension verified at {Path(args.paywall_extension_dir).resolve()}")


def summarize_text(
    text: str,
    *,
    checkpoint: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Summarization requires `transformers` and `torch`. Install them with "
            "`uv add transformers torch` (or pip) and try again."
        ) from exc

    resolved_device = device
    if device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(resolved_device)

    prompt = (
        "Summarize the following content in 3-5 concise bullet points. "
        "Avoid URLs and boilerplate.\n\n"
        f"{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(resolved_device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    generated_tokens = outputs[0][inputs.shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        run_init(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Convert HTML or a live URL to Markdown.")
    parser.add_argument("url", nargs="?", help="URL to scrape via patchright chromium.")
    parser.add_argument(
        "--browser-data-dir",
        default=".patchright-profile",
        help="Directory that Patchright reuses for persistent Chromium state.",
    )
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run the browser without a visible window (default).",
    )
    headless_group.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run the browser with a visible window.",
    )
    parser.add_argument(
        "--enable-paywall-bypass",
        action="store_true",
        help="Load the bundled Bypass Paywalls Clean extension before scraping.",
    )
    parser.add_argument(
        "--paywall-extension-dir",
        default="extensions/bypass-paywalls-chrome-clean",
        help="Path to the unpacked bypass extension assets when paywall bypassing is enabled.",
    )
    parser.add_argument(
        "--keep-urls",
        action="store_true",
        help="Preserve HTTP/HTTPS URLs in the markdown output instead of removing them.",
    )
    parser.add_argument(
        "--wait-until",
        default="domcontentloaded",
        choices=["domcontentloaded", "load", "networkidle"],
        help="Playwright wait condition for navigation (default: domcontentloaded).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Navigation timeout in milliseconds (default: 30000).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Output a short summary instead of the full markdown.",
    )
    parser.add_argument(
        "--summary-model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Model checkpoint to use for summarization.",
    )
    parser.add_argument(
        "--summary-device",
        default="cpu",
        help="Device for summarization model (cpu or cuda).",
    )
    parser.add_argument(
        "--summary-max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for the summary.",
    )
    parser.add_argument(
        "--summary-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the summary generation.",
    )
    parser.add_argument(
        "--summary-top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling for the summary generation.",
    )

    args = parser.parse_args()
    if args.url:
        bypass_extension_dir = Path(args.paywall_extension_dir)
        html = fetch_page_html(
            args.url,
            user_data_dir=args.browser_data_dir,
            channel="chrome",
            headless=args.headless,
            bypass_extension_dir=bypass_extension_dir if args.enable_paywall_bypass else None,
            wait_until=args.wait_until,
            timeout_ms=args.timeout_ms,
        )
    else:
        html = "..."

    markdown = md(html, strip_urls=not args.keep_urls)
    if args.summary_only:
        summary = summarize_text(
            markdown,
            checkpoint=args.summary_model,
            device=args.summary_device,
            max_new_tokens=args.summary_max_new_tokens,
            temperature=args.summary_temperature,
            top_p=args.summary_top_p,
        )
        print(summary)
        return

    print(markdown)


if __name__=='__main__':
    cli()
