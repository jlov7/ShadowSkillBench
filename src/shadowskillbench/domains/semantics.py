from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from shadowskillbench.domains.access import semantics as access_semantics
from shadowskillbench.domains.finance import semantics as finance_semantics
from shadowskillbench.skills.contamination import SkillSemanticRules

type Gate2Domain = Literal["access_provisioning", "financial_adjustments"]


@runtime_checkable
class DomainSkillSemanticsPlugin(Protocol):
    def skill_semantics(self) -> SkillSemanticRules: ...


class _FunctionSkillSemanticsPlugin:
    __slots__ = ("_factory",)

    def __init__(self, factory: object) -> None:
        if not callable(factory):
            raise TypeError("semantic factory must be callable")
        object.__setattr__(self, "_factory", factory)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("semantic plugin is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("semantic plugin is immutable")

    def skill_semantics(self) -> SkillSemanticRules:
        result = self._factory()
        return SkillSemanticRules.model_validate(result)


def get_domain_skill_semantics_plugin(domain: Gate2Domain) -> DomainSkillSemanticsPlugin:
    if type(domain) is not str:
        raise ValueError("GATE2_DOMAIN_INVALID")
    if domain == "access_provisioning":
        return _FunctionSkillSemanticsPlugin(access_semantics.skill_semantics)
    if domain == "financial_adjustments":
        return _FunctionSkillSemanticsPlugin(finance_semantics.skill_semantics)
    raise ValueError("GATE2_DOMAIN_INVALID")


def domain_skill_semantics(domain: Gate2Domain) -> SkillSemanticRules:
    plugin = get_domain_skill_semantics_plugin(domain)
    rules = SkillSemanticRules.model_validate(plugin.skill_semantics())
    if rules.domain != domain:
        raise ValueError("GATE2_DOMAIN_INVALID")
    return rules


__all__ = [
    "Gate2Domain",
    "DomainSkillSemanticsPlugin",
    "get_domain_skill_semantics_plugin",
    "domain_skill_semantics",
]
