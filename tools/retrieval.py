"""Small BM25 search over the reference corpus, used to ground teacher prompts in real 1.21.11 APIs.

    idx = ReferenceIndex.load(folder_with_jsonl, version="1.21.11")
    text = idx.context_for("a copper hammer that mines 3x3", budget_chars=40000)
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
IMPORT = re.compile(r"^import\s+(?:static\s+)?(net\.minecraft\.[\w.]+|com\.mojang\.[\w.]+)\s*;", re.M)


def tokenize(text: str) -> list[str]:
    # Splits camelCase too, so "BlockEntityRenderer" matches "block entity renderer".
    return [t.lower() for t in TOKEN.findall(text)]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def top(self, query: str, k: int) -> list[int]:
        terms = set(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]


class ReferenceIndex:
    def __init__(self, guides: list[dict], signatures: dict[str, str]):
        self.guides = guides  # docs + example code + Fabric API (non-deprecated)
        self.signatures = signatures  # fully-qualified class name -> javap text
        self.bm25 = BM25([f"{r['title']} {r['path']} {r['text']}" for r in guides])

    @classmethod
    def load(cls, folder: str | Path, version: str) -> "ReferenceIndex":
        folder = Path(folder)
        guides, signatures = [], {}
        for name in [f"docs-{version}.jsonl", f"fabric-api-{version}.jsonl"]:
            for line in open(folder / name, encoding="utf-8"):
                r = json.loads(line)
                if r["kind"] != "api-deprecated":
                    guides.append(r)
        for line in open(folder / f"minecraft-{version}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            signatures[r["path"]] = r["text"]
        return cls(guides, signatures)

    def signature(self, cls_name: str, max_lines: int = 60) -> str | None:
        text = self.signatures.get(cls_name)
        if text is None:
            return None
        lines = text.splitlines()
        return "\n".join(lines[:max_lines] + (["  ..."] if len(lines) > max_lines else []))

    def context_for(self, query: str, budget_chars: int = 40000, k: int = 8, max_signatures: int = 10) -> str:
        """Top guides/examples for the query, then signatures of the Minecraft classes those examples import."""
        parts, used = [], 0
        imported: list[str] = []
        for i in self.bm25.top(query, k):
            r = self.guides[i]
            chunk = f"--- {r['kind']}: {r['path']} ({r['title']}) ---\n{r['text'].strip()}\n"
            if used + len(chunk) > budget_chars * 0.75:
                continue
            parts.append(chunk)
            used += len(chunk)
            imported += [c for c in IMPORT.findall(r["text"]) if c not in imported]

        sigs = []
        for cls_name in imported:
            sig = self.signature(cls_name)
            if sig and used + len(sig) < budget_chars:
                sigs.append(sig)
                used += len(sig)
            if len(sigs) >= max_signatures:
                break
        if sigs:
            parts.append("--- exact Minecraft {} signatures (Mojang names) ---\n".format("class") + "\n\n".join(sigs))
        return "\n".join(parts)
