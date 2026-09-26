"""Loads restrictions.txt and runs a first-pass scan of mod code for harmful patterns.

The scan is a cheap heuristic filter for training data. It flags code that looks like
malware or info-leaking so it gets thrown out. The Reviewer specialist does the deeper check.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESTRICTIONS_FILE = ROOT / "restrictions.txt"

# Each pattern maps to the kind of harm it usually indicates.
SUSPICIOUS_PATTERNS = {
    "session/account token access": r"getAccessToken\s*\(|\.getUser\(\)\s*\.\s*getAccessToken|accessToken\b",
    "discord token or webhook": r"discord(app)?\.com/api/webhooks|Local Storage[/\\]+leveldb|\bmfa\.[\w-]{20,}",
    "browser credential files": r"Login Data|Cookies\b.*(Chrome|Edge|Brave|Opera)|\\\\Google\\\\Chrome\\\\User Data",
    "running external programs": r"Runtime\.getRuntime\(\)\s*\.\s*exec|new\s+ProcessBuilder\s*\(",
    "loading remote code": r"URLClassLoader|defineClass\s*\(|\.newInstance\(\).*http",
    "encoded payloads": r"Base64\.getDecoder\(\)\.decode\(\s*\"[A-Za-z0-9+/=]{200,}\"",
    "public IP lookup": r"api\.ipify\.org|checkip\.amazonaws|ip-api\.com|ifconfig\.me",
}


def load_restrictions(path: Path = RESTRICTIONS_FILE) -> list[str]:
    """Return every non-blank, non-comment line of restrictions.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def restrictions_prompt() -> str:
    """Restrictions formatted for a model prompt."""
    rules = "\n".join(f"- {rule}" for rule in load_restrictions())
    return (
        "You must refuse any request that matches one of these restrictions. "
        "Everything not listed is allowed.\n" + rules
    )


def scan_text(text: str) -> list[str]:
    """Return the names of suspicious patterns found in text."""
    return [name for name, pattern in SUSPICIOUS_PATTERNS.items() if re.search(pattern, text, re.IGNORECASE)]


def scan_dir(directory: Path) -> dict[str, list[str]]:
    """Scan every source/resource file in a directory. Returns {relative_path: [flags]}."""
    hits = {}
    for file in directory.rglob("*"):
        if file.is_file() and file.suffix in {".java", ".json", ".kt", ".gradle", ".properties", ".txt"}:
            flags = scan_text(file.read_text(encoding="utf-8", errors="ignore"))
            if flags:
                hits[str(file.relative_to(directory))] = flags
    return hits


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(scan_dir(Path(sys.argv[1])) or "clean")
    else:
        print(restrictions_prompt())
