"""Checks that go beyond compiling: boot a real dedicated server with the mod, then validate its assets.

  run_server(work)            -> (ok, errors, probe)  starts `gradlew runServer` with the probe mod installed;
                                 the probe dumps registered ids to run/probe.json and stops the server.
  validate_assets(files, probe) -> list of problems (missing blockstates/models/item definitions/lang entries)
"""
import json
import os
import re
import shutil
from pathlib import Path

from proc import run_capped

SERVER_PROPERTIES = """online-mode=false
level-type=minecraft\\:flat
generate-structures=false
spawn-protection=0
view-distance=2
simulation-distance=2
sync-chunk-writes=false
"""
PROBLEM_LINE = re.compile(r"(Exception|Error:|Caused by|FATAL|Mixin apply .* failed|Could not execute entrypoint|Incompatible mods)")
# Data folders Minecraft 1.21 renamed to singular; files in the old folders are silently ignored by the game.
OLD_DATA_DIRS = {"recipes": "recipe", "loot_tables": "loot_table", "advancements": "advancement",
                 "functions": "function", "structures": "structure", "predicates": "predicate",
                 "item_modifiers": "item_modifier", "tags/items": "tags/item", "tags/blocks": "tags/block",
                 "tags/entity_types": "tags/entity_type", "tags/fluids": "tags/fluid",
                 "tags/game_events": "tags/game_event", "tags/functions": "tags/function"}
NOISE = re.compile(r"(Environment: Environment|Failed to (retrieve|fetch) profile|Couldn't load server icon|com\.mojang\.authlib)")


def prepare_run_dir(work: Path, probe_jar: Path) -> Path:
    run = work / "run"
    shutil.rmtree(run / "world", ignore_errors=True)
    (run / "mods").mkdir(parents=True, exist_ok=True)
    shutil.copy2(probe_jar, run / "mods" / "nullified-probe.jar")
    (run / "eula.txt").write_text("eula=true\n")
    (run / "server.properties").write_text(SERVER_PROPERTIES)
    (run / "probe.json").unlink(missing_ok=True)
    return run


def extract_runtime_errors(output: str, limit_lines: int = 40) -> list[str]:
    """First few problem blocks (message + stack) from the server log."""
    lines = output.splitlines()
    errors, i = [], 0
    while i < len(lines) and len(errors) < 4:
        if PROBLEM_LINE.search(lines[i]) and not NOISE.search(lines[i]):
            block = lines[i:i + limit_lines]
            # keep the stack trace, stop at the next plain log line
            kept = [block[0]] + [l for l in block[1:] if l.strip().startswith(("at ", "Caused by", "...")) or "Exception" in l][:25]
            errors.append("\n".join(kept))
            i += len(kept)
        else:
            i += 1
    return errors


def run_server(work: Path, probe_jar: Path, gradlew: str = "./gradlew", timeout: int = 240) -> tuple[bool, list[str], dict]:
    run = prepare_run_dir(work, probe_jar)
    # run_capped kills the whole process group on timeout, so a mod that hangs the server can't
    # leave an orphaned JVM holding the output pipe open (which would block forever).
    cmd = [gradlew, "runServer", "--console=plain", "--no-daemon", "--args=nogui"]
    if os.name == "nt":
        cmd = ["cmd", "/c", *cmd]
    _returncode, output, timed_out = run_capped(cmd, work, timeout)

    # The probe mod writes probe.json and halts the server the moment startup completes, so if the
    # file exists the mod loaded successfully even if gradle was still shutting down at the timeout.
    probe_file = run / "probe.json"
    if probe_file.exists():
        return True, [], json.loads(probe_file.read_text())
    if timed_out:
        return False, extract_runtime_errors(output) or [f"server did not finish starting within {timeout}s"], {}
    return False, extract_runtime_errors(output) or [output[-3000:]], {}


def fix_data_dirs(files: dict[str, str]) -> dict[str, str]:
    """Move files out of data folders 1.21 renamed (recipes/ -> recipe/ ...). Purely mechanical, so no model needed."""
    out = {}
    for path, body in files.items():
        m = re.match(r"(src/main/resources/data/[^/]+/)(.+)$", path)
        if m:
            for old, new in OLD_DATA_DIRS.items():
                if m.group(2).startswith(old + "/"):
                    path = m.group(1) + new + m.group(2)[len(old):]
                    break
        out.setdefault(path, body)
    return out


def _models_in(node) -> list[str]:
    """All model ids referenced in a blockstate or item-model-definition JSON."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "model" and isinstance(value, str):
                found.append(value)
            else:
                found += _models_in(value)
    elif isinstance(node, list):
        for value in node:
            found += _models_in(value)
    return found


def validate_assets(files: dict[str, str], probe: dict) -> list[str]:
    """Problems that would show up in game as missing textures/models or untranslated names."""
    res = "src/main/resources/assets"
    problems = []

    def asset(ns: str, rel: str) -> str:
        return f"{res}/{ns}/{rel}"

    def load(path: str):
        try:
            return json.loads(files[path])
        except (KeyError, json.JSONDecodeError):
            return None

    def check_model(model_id: str, used_by: str):
        ns, _, path = model_id.partition(":") if ":" in model_id else ("minecraft", "", model_id)
        if ns == "minecraft":
            return
        mpath = asset(ns, f"models/{path}.json")
        if mpath not in files:
            problems.append(f"{used_by} references model {model_id}, but {mpath} does not exist")
            return
        model = load(mpath)
        parent = model.get("parent") if isinstance(model, dict) else None
        if isinstance(parent, str) and not parent.startswith(("minecraft:", "builtin/")) and ":" in parent:
            pns, _, ppath = parent.partition(":")
            if asset(pns, f"models/{ppath}.json") not in files:
                problems.append(f"model {model_id} has parent {parent}, which does not exist")

    lang = {}
    for path, body in files.items():
        if path.endswith("/lang/en_us.json"):
            try:
                lang.update(json.loads(body))
            except json.JSONDecodeError:
                problems.append(f"{path} is not valid JSON")

    block_item_of = dict(entry.split("=", 1) for entry in probe.get("block_items", []))
    for block_id in probe.get("blocks", []):
        ns, path = block_id.split(":", 1)
        bs_path = asset(ns, f"blockstates/{path}.json")
        bs = load(bs_path)
        if bs is None:
            problems.append(f"block {block_id} has no blockstate file {bs_path}")
        else:
            for m in dict.fromkeys(_models_in(bs)):
                check_model(m, f"blockstate {bs_path}")
        if f"block.{ns}.{path}" not in lang:
            problems.append(f"lang/en_us.json is missing \"block.{ns}.{path}\"")

    for item_id in probe.get("items", []):
        ns, path = item_id.split(":", 1)
        def_path = asset(ns, f"items/{path}.json")
        item_def = load(def_path)
        if item_def is None:
            problems.append(f"item {item_id} has no item model definition {def_path} "
                            f"(format: {{\"model\": {{\"type\": \"minecraft:model\", \"model\": \"{ns}:item/{path}\"}}}})")
        else:
            for m in dict.fromkeys(_models_in(item_def)):
                check_model(m, f"item definition {def_path}")
        lang_key = f"block.{ns}.{path}" if item_id in block_item_of else f"item.{ns}.{path}"
        if lang_key not in lang and f"item.{ns}.{path}" not in lang:
            problems.append(f"lang/en_us.json is missing \"{lang_key}\"")

    for path in files:
        m = re.match(r"src/main/resources/data/([^/]+)/(.+)$", path)
        if not m:
            continue
        for old, new in OLD_DATA_DIRS.items():
            if m.group(2).startswith(old + "/"):
                fixed = f"src/main/resources/data/{m.group(1)}/{new}/{m.group(2)[len(old) + 1:]}"
                problems.append(f"{path} is in the old folder data/{m.group(1)}/{old}/ which 1.21 ignores - "
                                f"move it to {fixed}")
                break
        if m.group(2).startswith(("recipe/", "loot_table/", "advancement/")) and path.endswith(".json"):
            try:
                json.loads(files[path])
            except json.JSONDecodeError as e:
                problems.append(f"{path} is not valid JSON: {e}")

    mod_json = load("src/main/resources/fabric.mod.json")
    if isinstance(mod_json, dict) and isinstance(mod_json.get("icon"), str) and f"src/main/resources/{mod_json['icon']}" not in files:
        problems.append(f"fabric.mod.json points to icon {mod_json['icon']}, which does not exist - remove the \"icon\" field")
    return list(dict.fromkeys(problems))
