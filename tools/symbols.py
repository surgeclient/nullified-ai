"""Lookups over the Minecraft signature index, used to repair compile errors and verify API facts.

    idx = SymbolIndex.load("reference/1.21.11/minecraft-1.21.11.jsonl")
    idx.find_class("BlockPos")          -> ["net.minecraft.core.BlockPos"]
    idx.methods("net.minecraft.world.item.Item$Properties", "sword")
"""
import json
import re
from collections import defaultdict
from pathlib import Path

# javac: "symbol:   class BlockPos" / "package net.foo does not exist" / "import a.b.C;" lines
SYMBOL = re.compile(r"symbol:\s+(class|method|variable)\s+(\w+)")
IMPORT_LINE = re.compile(r"import\s+(?:static\s+)?([\w.]+)\.(\w+)\s*;")
MISSING_PACKAGE = re.compile(r"package ([\w.]+) does not exist")


class SymbolIndex:
    def __init__(self, signatures: dict[str, str]):
        self.signatures = signatures
        self.by_simple = defaultdict(list)
        for fqn in signatures:
            self.by_simple[fqn.rsplit(".", 1)[-1].split("$")[-1]].append(fqn)

    @classmethod
    def load(cls, path: str | Path) -> "SymbolIndex":
        sigs = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            sigs[r["path"]] = r["text"]
        return cls(sigs)

    def find_class(self, simple: str) -> list[str]:
        return sorted(self.by_simple.get(simple, []), key=len)

    def methods(self, fqn: str, contains: str = "") -> list[str]:
        text = self.signatures.get(fqn, "")
        return [l.strip() for l in text.splitlines()[1:] if contains.lower() in l.lower()]

    def signature(self, fqn: str, max_lines: int = 50) -> str:
        lines = self.signatures.get(fqn, "").splitlines()
        return "\n".join(lines[:max_lines] + (["  ..."] if len(lines) > max_lines else []))

    def hints_for_errors(self, errors: list[str], files: dict[str, str], max_classes: int = 12) -> str:
        """Turn compile errors into concrete facts: where a class really lives, or that it doesn't exist,
        plus the real signatures of the Minecraft classes the code uses."""
        missing = set()
        for e in errors:
            for kind, name in SYMBOL.findall(e):
                if kind == "class":
                    missing.add(name)
            for _pkg, name in IMPORT_LINE.findall(e):
                if name[0].isupper():
                    missing.add(name)
        facts = []
        for name in sorted(missing):
            found = self.find_class(name)
            if found:
                facts.append(f"- {name} is {', '.join(found[:3])}")
            else:
                facts.append(f"- {name} does NOT exist in this Minecraft version; use the modern replacement")
        for e in errors:
            m = MISSING_PACKAGE.search(e)
            if m and not m.group(1).startswith("net.minecraft"):
                facts.append(f"- package {m.group(1)} does not exist")

        # Real signatures for Minecraft classes the code imports (that do exist), so wrong methods get fixed.
        used = []
        for body in files.values():
            for pkg, name in IMPORT_LINE.findall(body):
                fqn = f"{pkg}.{name}"
                if fqn in self.signatures and fqn not in used:
                    used.append(fqn)
        sigs = [self.signature(f) for f in used[:max_classes]]
        out = "Facts from the real game code:\n" + "\n".join(dict.fromkeys(facts))
        if sigs:
            out += "\n\nSignatures of classes you use:\n" + "\n\n".join(sigs)
        return out
