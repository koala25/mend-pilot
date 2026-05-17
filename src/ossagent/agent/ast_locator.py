"""AST-based code location using tree-sitter-python."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
PARSER = Parser(PY_LANGUAGE)


@dataclass(frozen=True)
class SymbolHit:
    path: str
    symbol: str
    kind: str
    line_start: int
    line_end: int


def find_python_symbol(
    repo_path: Path,
    symbol_name: str,
    *,
    max_results: int = 10,
) -> list[SymbolHit]:
    """Find function/method/class definitions matching `symbol_name` across .py files."""
    results: list[SymbolHit] = []
    for py_file in repo_path.rglob("*.py"):
        if any(part.startswith(".") for part in py_file.parts):
            continue
        try:
            text_bytes = py_file.read_bytes()
        except OSError:
            continue
        tree = PARSER.parse(text_bytes)
        for hit in _walk_for_symbol(tree.root_node, text_bytes, symbol_name):
            rel = py_file.relative_to(repo_path)
            results.append(
                SymbolHit(
                    path=str(rel),
                    symbol=hit["symbol"],
                    kind=hit["kind"],
                    line_start=hit["line_start"],
                    line_end=hit["line_end"],
                )
            )
            if len(results) >= max_results:
                return results
    return results


def _walk_for_symbol(
    node: Any,
    src_bytes: bytes,
    target: str,
    class_ctx: str = "",
) -> Iterator[dict[str, Any]]:
    """Yield {symbol, kind, line_start, line_end} for matching defs."""
    for child in node.children:
        if child.type == "class_definition":
            name = _identifier_child(child, src_bytes)
            if name == target:
                yield {
                    "symbol": name,
                    "kind": "class",
                    "line_start": child.start_point[0] + 1,
                    "line_end": child.end_point[0] + 1,
                }
            yield from _walk_for_symbol(child, src_bytes, target, class_ctx=name or class_ctx)
        elif child.type == "function_definition":
            name = _identifier_child(child, src_bytes)
            qualified = f"{class_ctx}.{name}" if class_ctx else name
            if name == target or qualified == target:
                yield {
                    "symbol": qualified,
                    "kind": "method" if class_ctx else "function",
                    "line_start": child.start_point[0] + 1,
                    "line_end": child.end_point[0] + 1,
                }
            yield from _walk_for_symbol(child, src_bytes, target, class_ctx)
        else:
            yield from _walk_for_symbol(child, src_bytes, target, class_ctx)


def _identifier_child(def_node: Any, src_bytes: bytes) -> str | None:
    for c in def_node.children:
        if c.type == "identifier":
            return src_bytes[c.start_byte : c.end_byte].decode("utf-8", errors="replace")
    return None
