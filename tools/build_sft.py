"""Turn generator runs into chat training data for Nullified AI.

Two kinds of examples, both with the same system prompt the model gets at use time
(restrictions.txt is enforced there and by the code scanner in restrictions.py):
  build - request -> PLAN + every file (only mods that passed compile + server boot + asset checks)
  fix   - failing files + problems (+ facts from the game code) -> changed files, when that fix made progress
"""
import json

from prompts import FIX_PROMPT, STAGE_EXPLAINED, STUDENT_SOLVE_PROMPT, format_files, runtime_hints, system_prompt

STAGE_RANK = {"compile": 0, "runtime": 1, "assets": 2, "passed": 3}


def _role_of(path: str) -> str:
    p = path.lower()
    if p.endswith("fabric.mod.json"):
        return "mod metadata and entrypoints"
    if p.endswith(".mixins.json") or "mixin" in p:
        return "mixin configuration"
    if p.endswith("lang/en_us.json"):
        return "English translations"
    if "/models/" in p:
        return "model"
    if "/blockstates/" in p:
        return "blockstate"
    if "/items/" in p:
        return "item model definition"
    if "/recipe" in p:
        return "recipe"
    if "/loot_table" in p:
        return "loot table"
    if "/tags/" in p:
        return "tag"
    if p.endswith(".java"):
        return "Java class " + path.rsplit("/", 1)[-1][:-5]
    return "resource"


def synth_plan(files: dict, version: str) -> str:
    """Real collected mods have no stored plan; derive a faithful one from their files so build examples
    still teach the PLAN->files format the model is asked for at inference."""
    mod_id = ""
    try:
        mod_id = json.loads(files["src/main/resources/fabric.mod.json"]).get("id", "")
    except (KeyError, json.JSONDecodeError, AttributeError):
        pass
    lines = [f"mod id: {mod_id}" if mod_id else "mod id: (see fabric.mod.json)"]
    for path in files:
        lines.append(f"- {path}: {_role_of(path)}")
    return "\n".join(lines)


def build_examples(records: list[dict], version: str) -> list[dict]:
    system = system_prompt(version)
    out = []
    for r in records:
        if r.get("ok") and r["files"]:
            plan = r["plan"].strip() or synth_plan(r["files"], version)
            answer = f"PLAN:\n{plan}\n\n{format_files(r['files'])}"
            out.append({"kind": "build", "id": r["id"], "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": STUDENT_SOLVE_PROMPT.format(version=version, request=r["request"])},
                {"role": "assistant", "content": answer},
            ]})
    return out


def fix_examples(records: list[dict], version: str, symbols) -> list[dict]:
    """A fix counts if the next check got further (or passed) than the one it answered."""
    system = system_prompt(version)
    out = []
    for r in records:
        attempts = r.get("attempts", [])
        final = "passed" if r.get("ok") else r.get("stage")
        for i, a in enumerate(attempts):
            # Older runs didn't store the files before each fix or the failing stage; skip those attempts.
            if "files_before" not in a or "stage" not in a or not a["changed"]:
                continue
            after = attempts[i + 1].get("stage") if i + 1 < len(attempts) else final
            if STAGE_RANK.get(after, -1) <= STAGE_RANK.get(a["stage"], -1):
                continue
            errors = "\n\n".join(e[:600] for e in a["errors"][:15])
            hints = symbols.hints_for_errors(a["errors"], a["files_before"])[:20000] + runtime_hints(a["errors"])
            out.append({"kind": "fix", "id": f"{r['id']}-fix{a['round']}", "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": FIX_PROMPT.format(
                    version=version, stage=STAGE_EXPLAINED[a["stage"]], request=r["request"],
                    files=format_files(a["files_before"]), errors=errors, hints=hints)},
                {"role": "assistant", "content": format_files(a["changed"])},
            ]})
    return out


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
