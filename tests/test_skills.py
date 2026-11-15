"""Tests for Skills Layer Manifests and Registry Validity."""

from __future__ import annotations

import pytest
from mcp_servers.client import TOOL_TO_SERVER_MAP
from skills.loader import list_skills, load_skill_manifest
from skills.schema import SkillManifest


def test_load_skill_manifests() -> None:
    """Test loading specific skill manifests by name."""
    eda_skill = load_skill_manifest("run_eda")
    assert isinstance(eda_skill, SkillManifest)
    assert eda_skill.name == "run_eda"
    assert eda_skill.mcp_tools_called == ["load_dataset", "inspect_dataset", "run_sql"]
    assert "fraud_bool" in " ".join(eda_skill.constraints)

    baseline_skill = load_skill_manifest("train_and_evaluate_baseline")
    assert isinstance(baseline_skill, SkillManifest)
    assert baseline_skill.name == "train_and_evaluate_baseline"
    assert baseline_skill.mcp_tools_called == ["create_features", "train_model", "evaluate_model"]
    assert "leakage" in " ".join(baseline_skill.constraints).lower()


def test_list_skills() -> None:
    """Test discovering and listing all skill manifests."""
    skills = list_skills()
    skill_names = {s.name for s in skills}
    assert "run_eda" in skill_names
    assert "train_and_evaluate_baseline" in skill_names
    assert len(skills) >= 2


def test_skill_manifest_tools_exist_in_mcp_registry() -> None:
    """Validity check: confirm every tool listed in mcp_tools_called exists in MCP routing map."""
    skills = list_skills()
    assert len(skills) > 0, "No skills found to validate"

    for skill in skills:
        for tool_name in skill.mcp_tools_called:
            assert tool_name in TOOL_TO_SERVER_MAP, (
                f"Skill '{skill.name}' references tool '{tool_name}' which is not registered in TOOL_TO_SERVER_MAP"
            )


def test_load_nonexistent_skill() -> None:
    """Test loading a non-existent skill manifest raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_skill_manifest("non_existent_skill_xyz")
