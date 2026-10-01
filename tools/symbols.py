"""Lookups over the Minecraft signature index, used to repair compile errors and verify API facts.

    idx = SymbolIndex.load("reference/1.21.11/minecraft-1.21.11.jsonl")
    idx.find_class("BlockPos")                              -> ["net.minecraft.core.BlockPos"]
    idx.methods("net.minecraft.world.item.Item$Properties", "sword")
    idx.suggest_methods("net.minecraft.world.item.Item", "finishUsing")   -> closest real methods, incl. inherited
"""
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

SYMBOL = re.compile(r"symbol:\s+(class|method|variable)\s+(\w+)")
LOCATION_TYPE = re.compile(r"location:\s+(?:variable \w+ of type|class|interface)\s+([\w.$<>]+)")
IMPORT_LINE = re.compile(r"import\s+(?:static\s+)?([\w.]+)\.(\w+)\s*;")
MISSING_PACKAGE = re.compile(r"package ([\w.]+) does not exist")
ERROR_HEAD = re.compile(r"(src/(?:main|client)/java/[\w/]+\.java):(\d+): error: (.*)")
HEADER = re.compile(r"(?:class|interface|enum|record)\s+([\w.$]+)(?:<.*?>)?(?:\s+extends\s+([\w.$<>?, ]+?))?(?:\s+implements\s+([\w.$<>?, ]+))?\s*\{")
METHOD = re.compile(r"\s([\w$]+)\((.*?)\)")
CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _strip_generics(t: str) -> str:
    depth, out = 0, []
    for ch in t:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


class SymbolIndex:
    def __init__(self, signatures: dict[str, str]):
        self.signatures = signatures
        self.by_simple = defaultdict(list)
        self.parents = {}
        for fqn, text in signatures.items():
            self.by_simple[fqn.rsplit(".", 1)[-1].split("$")[-1]].append(fqn)
            m = HEADER.search(text.splitlines()[0] + " {") if text else None
            if m:
                supers = [m.group(2) or "", m.group(3) or ""]
                self.parents[fqn] = [_strip_generics(s) for part in supers for s in part.split(",") if s.strip()]

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

    def find_constant(self, name: str) -> list[str]:
        """Classes declaring a public/protected static field with this name, e.g. BUILDING_BLOCKS -> CreativeModeTabs."""
        pat = re.compile(rf"static final [\w.$<>?, ]+ {re.escape(name)};")
        return sorted((f"{fqn}.{name}" for fqn, text in self.signatures.items() if pat.search(text)), key=len)

    def similar_constants(self, fqn: str, name: str, k: int = 8) -> list[str]:
        """Static fields of fqn closest to name, e.g. CreativeModeTabs.TOOLS -> TOOLS_AND_UTILITIES."""
        fields = re.findall(r"static final [\w.$<>?, ]+ ([A-Z][A-Z0-9_]*);", self.signatures.get(fqn, ""))
        tokens = set(name.lower().split("_"))
        def score(f):
            return difflib.SequenceMatcher(None, name.lower(), f.lower()).ratio() + 0.5 * len(tokens & set(f.lower().split("_")))
        return sorted(dict.fromkeys(fields), key=score, reverse=True)[:k]

    def public_methods(self, fqn: str, k: int = 25) -> list[str]:
        out = []
        for cls in self.hierarchy(fqn)[:3]:
            out += [f"{l.strip()}   [in {cls.rsplit('.', 1)[-1]}]" for l in self.signatures[cls].splitlines()[1:]
                    if l.strip().startswith("public") and METHOD.search(l)]
        return out[:k]

    def hierarchy(self, fqn: str, depth: int = 8) -> list[str]:
        seen, queue = [], [fqn]
        while queue and len(seen) < depth * 3:
            cur = queue.pop(0)
            if cur in seen or cur not in self.signatures:
                continue
            seen.append(cur)
            queue += self.parents.get(cur, [])
        return seen

    def suggest_methods(self, fqn: str, wanted: str, k: int = 6) -> list[str]:
        """Closest real methods to `wanted` in fqn and its supertypes (by name similarity)."""
        wanted_tokens = {t.lower() for t in CAMEL.findall(wanted)}
        scored = {}
        for cls in self.hierarchy(fqn):
            for line in self.signatures[cls].splitlines()[1:]:
                m = METHOD.search(line)
                if not m:
                    continue
                name = m.group(1)
                tokens = {t.lower() for t in CAMEL.findall(name)}
                score = difflib.SequenceMatcher(None, wanted.lower(), name.lower()).ratio() + 0.5 * len(wanted_tokens & tokens)
                key = f"{line.strip()}   [in {cls.rsplit('.', 1)[-1]}]"
                scored[key] = max(scored.get(key, 0), score)
        return [s for s, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]

    def resolve(self, simple: str, file_text: str) -> str | None:
        """Resolve a simple class name the way javac would: explicit import first, then unique match."""
        simple = _strip_generics(simple).split(".")[-1]
        for pkg, name in IMPORT_LINE.findall(file_text):
            if name == simple and f"{pkg}.{name}" in self.signatures:
                return f"{pkg}.{name}"
        found = self.find_class(simple)
        return found[0] if len(found) == 1 else None

    def hints_for_errors(self, errors: list[str], files: dict[str, str], max_chars: int = 20000) -> str:
        """Turn compile errors into concrete facts from the real game code."""
        facts = []
        for e in errors:
            head = ERROR_HEAD.search(e)
            path, lineno, msg = (head.group(1), int(head.group(2)), head.group(3)) if head else (None, 0, e)
            text = files.get(path, "")
            src_lines = text.splitlines()

            for kind, name in SYMBOL.findall(e):
                if kind == "variable" and name.isupper() and len(name) > 2:
                    owners = self.find_constant(name)
                    fact = f"- constant {name}: " + (", ".join(owners[:4]) if owners else "does NOT exist in this version")
                    used_on = re.search(rf"\b([A-Z]\w+)\.{re.escape(name)}\b", e)  # e.g. CreativeModeTabs.TOOLS
                    owner = self.resolve(used_on.group(1), text) if used_on else None
                    if owner:
                        close = self.similar_constants(owner, name)
                        if close:
                            fact += f". Real constants in {owner.rsplit('.', 1)[-1]} with similar names: " + ", ".join(close)
                    facts.append(fact)
                elif kind == "class" or (kind == "variable" and name[:1].isupper()):
                    found = self.find_class(name)
                    facts.append(f"- class {name}: " + (", ".join(found[:3]) if found else "does NOT exist in this version"))
                elif kind == "method":
                    loc = LOCATION_TYPE.search(e)
                    owner = self.resolve(loc.group(1), text) if loc else None
                    if owner is None and "super." in e:
                        ext = re.search(r"extends\s+(\w+)", text)
                        owner = self.resolve(ext.group(1), text) if ext else None
                    if owner:
                        facts.append(f"- {name}(...) does not exist on {owner.rsplit('.', 1)[-1]}. Closest real methods:\n    "
                                     + "\n    ".join(self.suggest_methods(owner, name)))

            if "does not override or implement a method from a supertype" in msg and src_lines:
                decl = " ".join(src_lines[lineno:lineno + 3])
                m = re.search(r"(\w+)\s*\(", decl)
                ext = re.search(r"extends\s+(\w+)", text)
                owner = self.resolve(ext.group(1), text) if ext else None
                if m and owner:
                    facts.append(f"- {m.group(1)}(...) is not a method of {owner.rsplit('.', 1)[-1]} or its parents. "
                                 f"Closest real methods you can override:\n    " + "\n    ".join(self.suggest_methods(owner, m.group(1))))

            prot = re.search(r"(\w+)\(.*?\) has (?:protected|private) access in ([\w.$]+)", msg)
            if prot:
                owner = self.resolve(prot.group(2), text) or (prot.group(2) if prot.group(2) in self.signatures else None)
                if owner:
                    facts.append(f"- {prot.group(1)}() can't be called from your code. Public methods of "
                                 f"{owner.rsplit('.', 1)[-1]} you can use instead:\n    " + "\n    ".join(self.public_methods(owner)))

            pm = MISSING_PACKAGE.search(msg)
            if pm:
                facts.append(f"- package {pm.group(1)} does not exist")
            for pkg, name in IMPORT_LINE.findall(e):
                if name[:1].isupper():
                    found = self.find_class(name)
                    facts.append(f"- import {pkg}.{name} is wrong: " + (f"use {found[0]}" if found else f"{name} does NOT exist"))

        out = "Facts from the real 1.21.11 game code:\n" + "\n".join(dict.fromkeys(facts))
        return out[:max_chars]
