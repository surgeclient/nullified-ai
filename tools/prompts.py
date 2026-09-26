"""Prompts for the teacher model and the parser that turns its answers into mod files.

The teacher answers in a fixed format so every answer can be split into
  PLAN  -> training data for the Planner specialist
  FILES -> training data for the Coder / Mixin / Assets specialists (and gets compile-checked)
"""
import re

from restrictions import restrictions_prompt

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
{restrictions}"""

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

FIX_PROMPT = """This Fabric mod for Minecraft {version} failed to compile.

<request>
{request}
</request>

<files>
{files}
</files>

<compiler_errors>
{errors}
</compiler_errors>

<reference>
{reference}
</reference>

Fix every error. Output ONLY the files you change, each COMPLETE, in this format:
=== FILE: <path> ===
<complete file contents>
=== END FILE ==="""

FILE_BLOCK = re.compile(r"^=== FILE: (?P<path>[^\n=]+?) ===\n(?P<body>.*?)^=== END FILE ===", re.S | re.M)
SAFE_PATH = re.compile(r"^src/(main|client)/(java|resources)/[\w./\-]+$")


def system_prompt(version: str) -> str:
    return SYSTEM.format(version=version, restrictions=restrictions_prompt())


def parse_answer(text: str) -> dict:
    """Split a teacher answer into plan, files and refusal. Unsafe paths are dropped."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    refused = re.match(r"^\s*REFUSED:\s*(.+)", text)
    if refused:
        return {"refused": refused.group(1).strip(), "plan": "", "files": {}, "dropped": []}
    plan = text.split("=== FILE:", 1)[0].replace("PLAN:", "", 1).strip()
    files, dropped = {}, []
    for m in FILE_BLOCK.finditer(text):
        path = m.group("path").strip()
        body = re.sub(r"^```\w*\n|```\s*$", "", m.group("body"), flags=re.M)  # strip stray code fences
        if SAFE_PATH.match(path) and ".." not in path:
            files[path] = body.rstrip() + "\n"
        else:
            dropped.append(path)
    return {"refused": None, "plan": plan, "files": files, "dropped": dropped}


def format_files(files: dict) -> str:
    return "\n".join(f"=== FILE: {p} ===\n{body}=== END FILE ===" for p, body in files.items())
