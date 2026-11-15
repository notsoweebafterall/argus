"""Skill Manifest loader for Argus ML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from skills.schema import SkillManifest

SKILLS_DIR = Path(__file__).parent


def load_skill_manifest(path_or_name: Union[str, Path]) -> SkillManifest:
    """Load and parse a SkillManifest by name or file path."""
    if isinstance(path_or_name, str):
        if path_or_name.endswith(".json") or path_or_name.endswith(".yaml") or path_or_name.endswith(".yml"):
            file_path = Path(path_or_name)
            if not file_path.is_absolute():
                file_path = SKILLS_DIR / file_path
        else:
            file_path = SKILLS_DIR / f"{path_or_name}.json"
    else:
        file_path = path_or_name

    if not file_path.exists():
        raise FileNotFoundError(f"Skill manifest file not found: '{file_path}'")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SkillManifest(**data)


def list_skills() -> list[SkillManifest]:
    """Discover and load all skill manifests in the skills directory."""
    manifests = []
    for json_file in sorted(SKILLS_DIR.glob("*.json")):
        try:
            manifests.append(load_skill_manifest(json_file))
        except Exception:
            pass
    return manifests
