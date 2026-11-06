#!/usr/bin/env python3
"""Flatten the main algo into the one .py file Optibook takes.

    python scripts/make_single_file.py --event config/event.json --out dist/optibook_bot.py

Modules are concatenated in dependency order, with their comments and docstrings kept as section
headers, so the result reads as a program rather than a blob. A flat file shares one namespace, so the
build fails on any name two modules define differently, rather than let one silently replace the other.
The output has only standard-library imports, except the organisers' client, imported at connect time.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "otk"
ENTRY = "main"
# Assigned in several modules with the same purpose. Logging under one name loses nothing.
SHARED_NAMES = {"logger"}
EVENT_TEMPLATE = """\
# ======================================================================================
# EDIT AFTER THE BRIEFING. All optional: everything else is read from the exchange.
#   position_limit, outstanding_limit : if the briefing changes them from 100 and 200
#   pairs            : [["LIQUID_NAME", "OTHER_NAME"]] if detection gets a pair wrong
#   index_weights    : {"ASML": 908.06, ...} to price the ETF and index products
#   option_vol       : volatility for option theoretical values
#   news_training_csv: path to the labelled headlines, if the event provides them
#   llm_url, llm_model, llm_key_env, llm_style ("openai" or "anthropic"): a provided LLM
#   params           : any runtime parameter, for example {"size": 10}
#   self_tune        : true: with no token, the bot tunes itself until update's freeze_at
# ======================================================================================
true, false, null = True, False, None  # so this block can hold JSON as it is
EVENT = %s
"""


@dataclass(frozen=True)
class Module:
    name: str
    tree: ast.Module
    lines: tuple[str, ...]


def load_module(name: str) -> Module:
    text = (SOURCE / f"{name}.py").read_text()
    return Module(name, ast.parse(text), tuple(text.splitlines()))


def local_imports(module: Module) -> set[str]:
    """Modules of ours imported at the top level. Imports inside functions stay lazy and stay there."""
    found = set()
    for node in module.tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("otk."):
            found.add(node.module.split(".", 1)[1])
    return found


def dependency_order(entry: str) -> list[Module]:
    ordered: list[Module] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        module = load_module(name)
        for dependency in sorted(local_imports(module)):
            visit(dependency)
        ordered.append(module)

    visit(entry)
    return ordered


def _is_dunder_all(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )


def _defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def check_collisions(modules: list[Module]) -> None:
    """Two modules may define a name only if the definitions are identical."""
    seen: dict[str, tuple[str, str]] = {}
    problems = []
    for module in modules:
        for node in module.tree.body:
            for name in _defined_names(node):
                if name in SHARED_NAMES:
                    continue
                text = "\n".join(module.lines[node.lineno - 1 : node.end_lineno])
                if name in seen and seen[name][1] != text:
                    problems.append(f"{name}: {seen[name][0]} and {module.name}")
                seen.setdefault(name, (module.name, text))
    if problems:
        raise SystemExit("name collisions, rename one of each:\n  " + "\n  ".join(sorted(problems)))


def split_module(module: Module) -> tuple[list[str], str]:
    """(top-level import lines to hoist, the rest of the module's text)."""
    drop: set[int] = set()
    imports: list[str] = []
    body = module.tree.body
    docstring = ast.get_docstring(module.tree, clean=True)
    for index, node in enumerate(body):
        lines = range(node.lineno - 1, node.end_lineno)
        if index == 0 and docstring is not None:
            drop.update(lines)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(lines)
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("otk"):
                aliased = [alias.name for alias in node.names if alias.asname]
                if aliased:
                    raise SystemExit(f"{module.name}: flatten cannot follow 'as' on {aliased}")
                continue
            imports.append("\n".join(module.lines[node.lineno - 1 : node.end_lineno]))
        elif _is_dunder_all(node):
            drop.update(lines)
    kept = [line for number, line in enumerate(module.lines) if number not in drop]
    header = [f"# {'=' * 96}", f"# {module.name}"]
    if docstring:
        header.extend(f"# {line}".rstrip() for line in docstring.splitlines())
    header.append(f"# {'=' * 96}")
    return imports, "\n".join(header) + "\n" + "\n".join(kept).strip() + "\n"


def flatten(event: dict | None = None, entry: str = ENTRY) -> str:
    modules = dependency_order(entry)
    check_collisions(modules)
    hoisted: list[str] = []
    sections: list[str] = []
    for module in modules:
        imports, section = split_module(module)
        hoisted.extend(item for item in imports if item not in hoisted)
        sections.append(section)
    parts = [
        "#!/usr/bin/env python3",
        '"""Optibook main algo, flattened from the team repo by scripts/make_single_file.py. Do not edit below EVENT."""',
        "",
        "from __future__ import annotations",
        "",
        *sorted(hoisted),
        "",
        EVENT_TEMPLATE % json.dumps(event or {}, indent=4),
        *sections,
        "",
        'if __name__ == "__main__":',
        "    # A notebook cell has no __file__, and its own arguments on sys.argv. It also means no parent to restart a crash.",
        '    _entry = globals().get("__file__")',
        "    sys.exit(main([] if _entry is None else None, event=EVENT, entry=_entry))",
        "",
    ]
    return "\n".join(parts)


def combine_events(layers) -> dict:
    """Left to right, last wins, and objects merge, so a preset's `params` and your own both stay."""
    merged: dict = {}
    for layer in layers:
        for key, value in layer.items():
            merged[key] = {**merged[key], **value} if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def build(output: Path, event: dict | None = None) -> Path:
    text = flatten(event)
    compile(text, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/optibook_bot.py")
    parser.add_argument("--event", action="append", default=[], help="event config JSON to embed in the EVENT block. Give several to combine them, last wins")
    args = parser.parse_args(argv)
    event = combine_events(json.loads(Path(path).read_text()) for path in args.event) or None
    path = build(Path(args.out), event)
    print(f"{path}: {len(path.read_text().splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
