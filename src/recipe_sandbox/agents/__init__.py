"""Unified agent implementations used across Recipe Sandbox.

This module uses lazy exports so subpackages can depend on concrete agent
modules without triggering circular imports during package initialization.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentMapper",
    "LLMClient",
    "build_mapping_prompt",
    "compile_mapping_fn",
    "extract_code_block",
    "load_schema_config",
    "validate_fn_on_samples",
    "validate_mapping_output",
]


def __getattr__(name: str):
    if name == "LLMClient":
        return getattr(import_module("recipe_sandbox.agents.base"), name)
    if name in {
        "AgentMapper",
        "build_mapping_prompt",
        "compile_mapping_fn",
        "extract_code_block",
        "load_schema_config",
        "validate_fn_on_samples",
        "validate_mapping_output",
    }:
        return getattr(import_module("recipe_sandbox.agents.schema_mapping"), name)
    raise AttributeError(f"module 'recipe_sandbox.agents' has no attribute {name!r}")
