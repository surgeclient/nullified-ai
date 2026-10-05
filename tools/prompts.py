"""Prompts for the teacher model and the parser that turns its answers into mod files.

The teacher answers in a fixed format so every answer can be split into
  PLAN  -> training data for the Planner specialist
  FILES -> training data for the Coder / Mixin / Assets specialists (and gets compile-checked)
"""
import re
from pathlib import Path

from restrictions import load_restrictions, restrictions_prompt

CHEATSHEETS = Path(__file__).resolve().parent / "cheatsheet"

# Topics come from the Fabric docs sections; each has a few example ideas to steer variety.
TOPICS = {
    "items": "custom items, tools, armor, food, item components, custom item behavior on use",
    "blocks": "custom blocks, block states, block properties, redstone behavior, block interaction",
    "block-entities": "block entities that store data, tick, or have an inventory/menu",
    "entities": "custom mobs and entities, attributes, AI goals, spawning, entity renderers and models",
    "commands": "Brigadier commands with arguments, permissions, suggestions and feedback",
    "events": "Fabric API event callbacks (player, block, entity, server, world, tick events)",
    "networking": "custom payloads between client and server, syncing data",
    "mixins": "mixins that inject into or modify vanilla behavior (Inject, ModifyReturnValue, Redirect, accessors/invokers)",
    "rendering": "HUD elements, world rendering, custom screens and GUI widgets",
    "data-generation": "datagen providers for recipes, loot tables, tags, models and language files",
    "recipes": "custom recipes and recipe types",
    "sounds": "custom sound events and playing sounds",
    "key-mappings": "client key bindings that trigger actions",
    "enchantments": "custom enchantments and enchantment effects (data-driven)",
    "effects": "custom status effects and potions",
    "game-rules": "custom game rules and config options",
    "particles": "custom particles and particle effects",
    "world-gen": "ores, features, biome modifications",
    "client-utility": "client-side quality-of-life features: zoom, info overlays, toggles, keybind helpers",
}

DIFFICULTY = {
    1: "small: 1-3 Java classes, one feature",
    2: "medium: 3-6 classes, a couple of connected features",
    3: "large: 6-12 classes, a small but complete mod with several systems working together",
}

SYSTEM = """You are an expert Fabric mod developer for Minecraft {version}.
Rules:
- Target Minecraft {version}, Fabric Loader 0.19.5, Fabric API, Java 21, OFFICIAL MOJANG MAPPINGS.
- Use ONLY APIs that exist in {version}. The reference material below is the source of truth; prefer it over memory.
  Example: in {version} the id class is net.minecraft.resources.Identifier (NOT ResourceLocation).
- Never use deprecated Fabric API modules.
- The project uses split source sets: common code in src/main/java, client-only code in src/client/java
  (client entrypoint, renderers, screens, key bindings, client mixins).
{restrictions}

{cheatsheet}"""

REQUEST_PROMPT = """Write {n} different, realistic requests that a Minecraft player or modder might send to an AI
that builds Fabric mods. Topic: {topic} ({topic_hint}). Size: {difficulty}.
Make them specific (names, numbers, behaviors) and varied in style: some casual, some detailed.
Do not number them with explanations - output ONLY a JSON list of strings."""

SOLVE_PROMPT = """Build this Fabric mod for Minecraft {version}:

<request>
{request}
</request>

<reference>
{reference}
</reference>

Answer in EXACTLY this format:

PLAN:
<mod id, then a short list of every file you will create and what it does>

=== FILE: <path starting with src/> ===
<complete file contents>
=== END FILE ===
(repeat for every file)

Requirements:
- Include src/main/resources/fabric.mod.json (schemaVersion 1, "version": "${{version}}", depends on
  fabricloader >=0.19.5, minecraft ~{version}, java >=21, fabric-api *), with correct entrypoints and mixin configs.
- Mod id: lowercase letters, digits, underscores. Java package: com.<author>.<modid>.
- Include every asset/data JSON the feature needs (item model definitions in assets/<modid>/items/,
  models, blockstates, lang/en_us.json, recipes, loot tables, tags).
- Every file complete - no placeholders, no "..." or TODOs.
- If the request matches a restriction, output only: REFUSED: <one-sentence reason>"""

# What Nullified AI itself is asked at build time: same format as the teacher, without the retrieved reference.
STUDENT_SOLVE_PROMPT = SOLVE_PROMPT.replace("""<reference>
{reference}
</reference>

""", "")

# Focused single-file training: teaches one correct {version} file at a time (short, so almost all real mods fit
# the training window). Many short, repeated lessons beat a few giant ones for a small model.
FILE_PROMPT = """Write one file of a working Fabric mod for Minecraft {version} (Mojang mappings).

<request>
{request}
</request>

<mod_metadata file="src/main/resources/fabric.mod.json">
{mod_json}
</mod_metadata>

Write EXACTLY this one file, complete and correct for {version}, in this format:
=== FILE: {path} ===
<complete file contents>
=== END FILE ==="""

STAGE_EXPLAINED = {
    "compile": "It failed to compile. The compiler errors are below.",
    "runtime": "It compiled, but it crashed or failed while a real 1.21.11 server was starting with it. The log is below.",
    "assets": "It compiled and ran, but its resource files are incomplete, so blocks/items would look broken in game.",
}

# Known runtime failures -> the fact that fixes them.
RUNTIME_FACTS = {
    "id not set": "Every Item.Properties needs .setId(itemKey) and every BlockBehaviour.Properties needs .setId(blockKey) "
                  "BEFORE the item/block is constructed.",
    "Mixin apply": "A mixin target method/signature is wrong. Check the exact method name and parameters in 1.21.11.",
    "Could not execute entrypoint": "An exception was thrown during mod initialization (see the 'Caused by' line).",
    "Duplicate registration": "Something was registered twice under the same id.",
}


def runtime_hints(errors: list[str]) -> str:
    text = "\n".join(errors).lower()
    facts = [fact for key, fact in RUNTIME_FACTS.items() if key.lower() in text]
    return ("\n" + "\n".join(f"- {f}" for f in facts)) if facts else ""


# Removed/renamed APIs the base model reaches for out of old-Minecraft memory -> the exact 1.21.11 replacement.
# Each entry: list of trigger substrings (matched in the raw compile errors) -> one replacement instruction.
MIGRATIONS = [
    (["ResourceLocation"],
     "`ResourceLocation` is not the name here. Use `net.minecraft.resources.Identifier` and build ids with "
     "`Identifier.fromNamespaceAndPath(MOD_ID, path)` - there is no public `new ResourceLocation(...)` constructor."),
    (["TAB_COMBAT", "TAB_TOOLS", "TAB_MISC", "CreativeModeTab.TAB", ".tab(", "ItemGroup"],
     "Creative tabs are NOT set with `Item.Properties.tab(...)` or `CreativeModeTab.TAB_*` (both removed). Remove any "
     "`.tab(...)` call, and add the item to a tab with an event: "
     "`ItemGroupEvents.modifyEntriesEvent(CreativeModeTabs.COMBAT).register(e -> e.accept(THE_ITEM));` "
     "(import `net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents`; tab constants live on `CreativeModeTabs`)."),
    (["SwordItem", "PickaxeItem", "AxeItem", "ShovelItem", "HoeItem", "DiggerItem", "class Tier", "Tiers"],
     "`SwordItem`/`PickaxeItem`/`Tier`/`Tiers` do not exist. A tool is a plain `Item` built with a tool property: "
     "`new Item(new Item.Properties().setId(key).sword(ToolMaterial.IRON, 3f, -2.4f))` "
     "(also `.pickaxe/.axe/.shovel/.hoe`). Material constants: `ToolMaterial.WOOD/STONE/IRON/DIAMOND/NETHERITE`."),
    (["CraftingManager", "ShapedRecipe", "ShapelessRecipe", "RecipeType.CRAFTING", "addRecipe"],
     "Do NOT create recipes in Java (`ShapedRecipe`/`CraftingManager` are not used). Recipes are JSON data files at "
     "`src/main/resources/data/<modid>/recipe/<name>.json` with type `minecraft:crafting_shaped` - delete the Java "
     "recipe code and ship the JSON file instead."),
    (["InteractionResultHolder"],
     "`InteractionResultHolder` does not exist. `Item.use(...)` returns a plain `InteractionResult`."),
]


def migration_hints(errors: list[str]) -> str:
    text = "\n".join(errors)
    facts = []
    for triggers, fix in MIGRATIONS:
        if any(t in text for t in triggers) and fix not in facts:
            facts.append(fix)
    return ("\n" + "\n".join(f"- {f}" for f in facts)) if facts else ""


FIX_PROMPT = """This Fabric mod for Minecraft {version} failed a check. {stage}

<request>
{request}
</request>

<files>
{files}
</files>

<problems>
{errors}
</problems>

<facts>
{hints}
</facts>

Fix every problem. Keep all of the mod's features - never delete a feature to make an error go away. Output ONLY the files you change, each COMPLETE, in this format:
=== FILE: <path> ===
<complete file contents>
=== END FILE ==="""

FILE_HEADER = re.compile(r"^=== FILE:\s*(?P<path>[^\s=][^=]*?)\s*(?:===)?\s*$")
END_MARKER = re.compile(r"^=== END FILE\s*(?:===)?\s*$")
SAFE_PATH = re.compile(r"^src/(main|client)/(java|resources)/[\w./\-]+$")


REVIEW_PROMPT = """You are the gatekeeper for a Minecraft mod builder. Decide whether this request clearly matches
one of the owner's restrictions.

<restrictions>
{rules}
</restrictions>

<request>
{request}
</request>

Only block a request that clearly matches a restriction. Everything else is allowed, including admin, moderation,
teleport and spawn commands, PvP and combat features, client-side utilities, and anything else that is normal
Minecraft modding.
Answer with exactly one line: ALLOWED, or BLOCKED: <the restriction it matches>"""


def review_prompt(request: str) -> str:
    return REVIEW_PROMPT.format(rules="\n".join(f"- {r}" for r in load_restrictions()), request=request)


def parse_review(text: str) -> str | None:
    """The matched restriction if the gate blocked the request, else None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"BLOCKED:\s*(.+)", text)
    return m.group(1).strip() if m and not text.upper().startswith("ALLOWED") else None


def system_prompt(version: str) -> str:
    sheet = CHEATSHEETS / f"{version}.md"
    cheatsheet = sheet.read_text(encoding="utf-8") if sheet.exists() else ""
    return SYSTEM.format(version=version, restrictions=restrictions_prompt(), cheatsheet=cheatsheet)


def parse_answer(text: str) -> dict:
    """Split a teacher answer into plan, files and refusal. Unsafe paths are dropped."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    refused = re.match(r"^\s*REFUSED:\s*(.+)", text)
    if refused:
        return {"refused": refused.group(1).strip(), "plan": "", "files": {}, "dropped": []}
    plan = text.split("=== FILE:", 1)[0].replace("PLAN:", "", 1).strip()
    # Line-based and lenient: models sometimes write "=== END FILE" without the closing "===",
    # or skip the end marker and start the next file directly.
    blocks, current = [], None
    for line in text.splitlines():
        header = FILE_HEADER.match(line)
        if header:
            if current:
                blocks.append(current)
            current = (header.group("path").strip(), [])
        elif END_MARKER.match(line):
            if current:
                blocks.append(current)
            current = None
        elif current:
            current[1].append(line)
    if current:
        blocks.append(current)

    files, dropped = {}, []
    for path, lines in blocks:
        body = re.sub(r"^```\w*\n|```\s*$", "", "\n".join(lines), flags=re.M)  # strip stray code fences
        if path.endswith(".java"):
            body = dedupe_imports(body)
        if SAFE_PATH.match(path) and ".." not in path:
            files[path] = body.rstrip() + "\n"
        else:
            dropped.append(path)
    return {"refused": None, "plan": plan, "files": files, "dropped": dropped}


def dedupe_imports(java: str) -> str:
    """Drop repeated identical import lines (a symptom of the model getting stuck in a loop)."""
    seen, out = set(), []
    for line in java.splitlines():
        key = line.strip()
        if key.startswith("import "):
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def looks_looped(text: str, threshold: int = 25) -> bool:
    """True if one non-trivial line repeats many times - the answer is stuck and should be resampled."""
    counts = {}
    for line in text.splitlines():
        key = line.strip()
        if len(key) > 12:
            counts[key] = counts.get(key, 0) + 1
    return bool(counts) and max(counts.values()) >= threshold


def format_files(files: dict) -> str:
    return "\n".join(f"=== FILE: {p} ===\n{body}=== END FILE ===" for p, body in files.items())
