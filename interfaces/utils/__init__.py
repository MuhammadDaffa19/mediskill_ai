# interfaces/utils/__init__.py
"""
Utility package for interface routing.

Exports:
- load_json
- detect_intent, extract_keywords
- choose_interfaces
"""
from .json_loader import load_json
from .intent_rules import detect_intent, extract_keywords
from .interface_router import choose_interfaces

__all__ = [
    "load_json",
    "detect_intent",
    "extract_keywords",
    "choose_interfaces",
]
