from __future__ import annotations

from shadowskillbench.core.hashing import canonical_json_bytes, sha256_ref
from shadowskillbench.skills.models import SkillIR, parse_skill_ir

_PROFILE = "SSB-SKILLMD1"


def _escape_markdown(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "[", "]", "<", ">", "#", "|"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _lines_for_list(items: tuple[str, ...]) -> list[str]:
    return [f"- {_escape_markdown(item)}" for item in items] or ["- None."]


def _render(parsed: SkillIR) -> str:
    lines = [
        f"# {_escape_markdown(parsed.skill_id)}",
        "",
        f"Domain: {_escape_markdown(parsed.domain)}",
        "",
        _escape_markdown(parsed.objective),
    ]
    sections = (
        ("When to use", parsed.applicability),
        ("Required inputs", parsed.required_inputs),
        ("Preconditions", parsed.preconditions),
    )
    for heading, items in sections:
        lines.extend(("", f"## {heading}", "", *_lines_for_list(items)))
    lines.extend(("", "## Steps"))
    for index, step in enumerate(parsed.ordered_steps, start=1):
        arguments = canonical_json_bytes(step.argument_bindings).decode("utf-8")
        lines.extend(
            (
                "",
                f"### {index}. {_escape_markdown(step.action_intent)}",
                "",
                f"Tool: {_escape_markdown(step.tool_name)}",
                f"Optional: {'yes' if step.optional else 'no'}",
                "Preconditions:",
                *_lines_for_list(step.preconditions),
                "Arguments:",
                f"    {arguments}",
            )
        )
    for heading, items in (
        ("Decision hints", parsed.decision_hints),
        ("Verification", parsed.verification_steps),
        ("Stop conditions", parsed.stop_conditions),
        ("Escalation", parsed.escalation_hints),
    ):
        lines.extend(("", f"## {heading}", "", *_lines_for_list(items)))
    return "\n".join(lines) + "\n"


def render_skill(value: object) -> str:
    return _render(parse_skill_ir(value))


def rendered_skill_bytes(value: object) -> bytes:
    return render_skill(value).encode("utf-8")


def hash_rendered_skill(value: object) -> str:
    rendered = render_skill(value)
    return sha256_ref({"profile": _PROFILE, "rendered_text": rendered})


__all__ = ["hash_rendered_skill", "render_skill", "rendered_skill_bytes"]
