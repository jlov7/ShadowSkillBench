from __future__ import annotations

# ruff: noqa: E501
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "compiler.py"
_TARGETED_ASSERTION_SENTINEL = "D026_TARGETED_ASSERTION_FAILED"
_TARGETED_ASSERTION_EXIT_CODE = 86


@pytest.mark.mutation
def test_compiler_source_has_no_ambient_or_provider_surface() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_modules = {
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "http",
        "httpx",
        "requests",
        "urllib",
        "random",
        "secrets",
        "time",
        "datetime",
        "uuid",
        "logging",
        "openai",
        "anthropic",
    }
    forbidden_calls = {
        "open",
        "read_text",
        "write_text",
        "unlink",
        "remove",
        "mkdir",
        "makedirs",
        "getenv",
        "putenv",
        "run",
        "Popen",
        "check_call",
        "check_output",
        "sleep",
        "monotonic",
        "perf_counter",
        "connect",
        "send",
        "recv",
        "get",
        "post",
        "request",
        "urlopen",
        "getLogger",
        "basicConfig",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert not imports & forbidden_modules
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    calls.update(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    assert not calls & forbidden_calls


@pytest.mark.mutation
def test_manifest_is_cycle_free_and_pre_call_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    manifest = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CompilerManifest"
    )
    fields = {
        node.target.id
        for node in manifest.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert fields == {
        "manifest_profile",
        "compiler_input_hash",
        "prompt_profile",
        "system_prompt_raw_hash",
        "request_profile",
        "instruction_provenance_profile",
        "structured_output_schema_hash",
        "declared_capabilities",
        "temperature",
        "seed",
        "max_tokens",
    }
    forbidden = {
        "request_envelope_hash",
        "raw_request_hash",
        "raw_response_hash",
        "usage",
        "cost",
        "output",
        "rendered_skill",
    }
    assert not fields & forbidden
    compile_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "compile_skill"
    )
    call_lines = [
        node.lineno
        for node in ast.walk(compile_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "structured"
    ]
    manifest_lines = [
        node.lineno
        for node in ast.walk(compile_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CompilerManifest"
    ]
    assert len(call_lines) == len(manifest_lines) == 1
    assert manifest_lines[0] < call_lines[0]


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "assertion"),
    [
        (
            "compiler_input_data = compiler_input_projection(compiler_input)",
            "compiler_input_data = compiler_input_projection(bundle)",
            "assert_projection_reaches_probe_and_sanitizes()",
        ),
        (
            "await client.structured(request, output_schema)",
            "await client.structured(request, object)",
            "assert_probe_schema()",
        ),
        (
            'Message(role="system", content=system_content)',
            'Message(role="developer", content=system_content)',
            "assert_probe_system_role()",
        ),
        (
            'Message(role="user", content=user_content)',
            'Message(role="developer", content=user_content)',
            "assert_probe_user_role()",
        ),
        (
            "compiler_input_hash = hash_compiler_input(compiler_input)",
            'compiler_input_hash = "sha256:" + "0" * 64',
            "assert_probe_input_hash()",
        ),
        (
            "not capabilities.supports_structured_output",
            "False",
            "assert_capability_rejected()",
        ),
        (
            "if parsed.compiler_manifest_ref != compiler_manifest_ref:",
            "if False:",
            "assert_manifest_rejected()",
        ),
        (
            'or provenance["compiler_input_hash"] != compiler_input_hash',
            "or False",
            "assert_provenance_rejected()",
        ),
        (
            'provenance["profile"] != _PROVENANCE_PROFILE',
            "False",
            "assert_provenance_profile_rejected()",
        ),
        (
            "self.structured_output_schema_hash\n            != self.compiler_manifest.structured_output_schema_hash",
            "False",
            "assert_direct_artifact_schema_rejected()",
        ),
        (
            "if self.skill_ir.compiler_manifest_ref != self.compiler_manifest_hash:",
            "if False:",
            "assert_direct_artifact_manifest_rejected()",
        ),
        (
            'or set(provenance) != {"profile", "compiler_input_hash", "instruction_evidence"}',
            "or False",
            "assert_direct_artifact_extra_key_rejected()",
        ),
        (
            'or type(provenance["instruction_evidence"]) is not dict',
            "or False",
            "assert_direct_artifact_evidence_type_rejected()",
        ),
        (
            'provenance["profile"] != self.compiler_manifest.instruction_provenance_profile',
            "False",
            "assert_direct_artifact_profile_rejected()",
        ),
        (
            'provenance["compiler_input_hash"] != self.compiler_manifest.compiler_input_hash',
            "False",
            "assert_direct_artifact_input_hash_rejected()",
        ),
        (
            "self.cost.input_nanos != self.usage.input_tokens * self.cost.input_nanos_per_token",
            "False",
            "assert_direct_artifact_cost_rejected()",
        ),
    ],
)
def test_compiler_mutants_are_killed(
    tmp_path: Path, needle: str, replacement: str, assertion: str
) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / "compiler_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    check.write_text(_CHECK.replace("TARGETED_ASSERTION_CALL", assertion), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(check), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == _TARGETED_ASSERTION_EXIT_CODE, (
        f"unexpected exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _TARGETED_ASSERTION_SENTINEL in result.stdout, result.stderr


_CHECK = """
import asyncio
import importlib.util
import json
import sys
from decimal import Decimal
from shadowskillbench.models import ProviderCapabilities, TokenCost, TokenUsage
from shadowskillbench.skills.models import SkillIR
from shadowskillbench.traces.bundles import generate_bundle
from pathlib import Path

spec = importlib.util.spec_from_file_location("shadowskillbench.skills.compiler_mutant", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class ProbeClient:
    def __init__(self):
        self.calls = []
        self.capabilities = ProviderCapabilities(
            provider="scripted", model="model-test", model_version="v1",
            supports_system_role=True, supports_developer_role=True,
            supports_seed=True, supports_structured_output=True,
        )

    async def structured(self, request, schema):
        self.calls.append((request, schema))
        raise RuntimeError("probe-stop")

def run():
    global compile_error, probe_client_calls, probe_request, probe_schema
    bundle = generate_bundle("access_provisioning", Decimal("0.5"), 12, 2401).bundle
    config = module.CompilerConfig(
        prompt_profile="SSB-SKILL-COMPILER1", prompt_bytes=b"compile\\n",
        temperature=0.0, seed=7, max_tokens=128,
    )
    client = ProbeClient()
    compile_error = None
    try:
        asyncio.run(module.compile_skill(bundle, client, config))
    except module.CompilerContractError as error:
        compile_error = error
    else:
        raise AssertionError("nonconforming client failure was not sanitized")
    probe_client_calls = client.calls
    if client.calls:
        request, schema = client.calls[0]
        probe_request = request
        probe_schema = schema
        envelope = json.loads(request.messages[1].content)
        assert envelope["request_profile"] == "SSB-SKILL-COMPILER-REQUEST1"
        assert set(envelope) == {"request_profile", "compiler_input", "required_output_bindings"}
        assert set(envelope["required_output_bindings"]) == {
            "compiler_manifest_ref", "compiler_input_hash", "instruction_provenance_profile",
        }
        assert envelope["required_output_bindings"]["instruction_provenance_profile"] == "SSB-INSTRUCTION-PROVENANCE1"
        assert "worker_policy_id" not in request.messages[1].content
        assert "compliant" not in request.messages[1].content
    try:
        TARGETED_ASSERTION_CALL
    except AssertionError:
        print("D026_TARGETED_ASSERTION_FAILED")
        raise SystemExit(86)

def assert_projection_reaches_probe_and_sanitizes():
    assert compile_error is not None
    assert compile_error.code == "COMPILER_OUTPUT_INVALID"
    assert probe_client_calls

def assert_probe_request():
    assert tuple(message.role for message in probe_request.messages) == ("system", "user")

def assert_probe_schema():
    assert probe_schema is SkillIR

def assert_probe_system_role():
    assert probe_request.messages[0].role == "system"

def assert_probe_user_role():
    assert probe_request.messages[1].role == "user"

def assert_probe_input_hash():
    envelope = json.loads(probe_request.messages[1].content)
    expected = module.sha256_ref(envelope["compiler_input"])
    assert envelope["required_output_bindings"]["compiler_input_hash"] == expected

def _config():
    return module.CompilerConfig(
        prompt_profile="SSB-SKILL-COMPILER1", prompt_bytes=b"compile\\n",
        temperature=0.0, seed=7, max_tokens=128,
    )

def assert_capability_rejected():
    capabilities = ProviderCapabilities(
        provider="scripted", model="model-test", model_version="v1",
        supports_system_role=True, supports_developer_role=True,
        supports_seed=True, supports_structured_output=False,
    )
    try:
        module._validate_declared_capabilities(capabilities, _config())
    except ValueError:
        return
    raise AssertionError("structured-output capability was not rejected")

def _binding_skill(input_value, input_hash, manifest_ref, *, wrong_manifest=False, wrong_input=False):
    event_id = input_value.traces[0].events[0].event_id
    provenance_hash = "sha256:" + "c" * 64 if wrong_input else input_hash
    skill_manifest = "sha256:" + "d" * 64 if wrong_manifest else manifest_ref
    return SkillIR.model_validate({
        "skill_id": "skill_access_provisioning", "schema_version": "1.0",
        "domain": "access_provisioning", "objective": "objective",
        "applicability": [], "required_inputs": [], "preconditions": [],
        "ordered_steps": [{"step_id": "step_0", "action_intent": "act",
            "tool_name": "tool", "argument_bindings": {}, "preconditions": [],
            "optional": False, "evidence_refs": [event_id]}],
        "decision_hints": [], "verification_steps": [], "stop_conditions": [],
        "escalation_hints": [], "source_trace_ids": [trace.trace_id for trace in input_value.traces],
        "instruction_provenance": {
            "profile": "SSB-INSTRUCTION-PROVENANCE1",
            "compiler_input_hash": provenance_hash,
            "instruction_evidence": {"/objective": [event_id], "/ordered_steps/0": [event_id]},
        },
        "compiler_manifest_ref": skill_manifest,
    })

def assert_manifest_rejected():
    bundle = generate_bundle("access_provisioning", Decimal("0.5"), 12, 2402).bundle
    value = module.compiler_view(bundle)
    digest = module.hash_compiler_input(value)
    skill = _binding_skill(value, digest, "sha256:" + "b" * 64, wrong_manifest=True)
    try:
        module._validate_output_bindings(skill, value, digest, "sha256:" + "b" * 64)
    except ValueError:
        return
    raise AssertionError("mismatched compiler manifest was not rejected")

def assert_provenance_rejected():
    bundle = generate_bundle("access_provisioning", Decimal("0.5"), 12, 2403).bundle
    value = module.compiler_view(bundle)
    digest = module.hash_compiler_input(value)
    skill = _binding_skill(value, digest, "sha256:" + "b" * 64, wrong_input=True)
    try:
        module._validate_output_bindings(skill, value, digest, "sha256:" + "b" * 64)
    except ValueError:
        return
    raise AssertionError("mismatched compiler-input provenance was not rejected")

def assert_provenance_profile_rejected():
    bundle = generate_bundle("access_provisioning", Decimal("0.5"), 12, 2404).bundle
    value = module.compiler_view(bundle)
    digest = module.hash_compiler_input(value)
    skill = _binding_skill(value, digest, "sha256:" + "b" * 64)
    raw = module.skill_ir_projection(skill)
    raw["instruction_provenance"]["profile"] = "SSB-WRONG1"
    skill = SkillIR.model_validate(raw)
    try:
        module._validate_output_bindings(skill, value, digest, "sha256:" + "b" * 64)
    except ValueError:
        return
    raise AssertionError("mismatched provenance profile was not rejected")

def _direct_artifact():
    caps = ProviderCapabilities(
        provider="scripted", model="model-test", model_version="v1",
        supports_system_role=True, supports_developer_role=True,
        supports_seed=True, supports_structured_output=True,
    )
    input_hash = "sha256:" + "a" * 64
    schema_hash = "sha256:" + "b" * 64
    manifest = module.CompilerManifest(
        manifest_profile="SSB-COMPILER-MANIFEST1", compiler_input_hash=input_hash,
        prompt_profile="SSB-SKILL-COMPILER1", system_prompt_raw_hash="sha256:" + "c" * 64,
        request_profile="SSB-SKILL-COMPILER-REQUEST1",
        instruction_provenance_profile="SSB-INSTRUCTION-PROVENANCE1",
        structured_output_schema_hash=schema_hash, declared_capabilities=caps,
        temperature=0.0, seed=7, max_tokens=128,
    )
    manifest_hash = module.compiler_manifest_hash(manifest)
    skill = SkillIR.model_validate({
        "skill_id": "skill_access_provisioning", "schema_version": "1.0",
        "domain": "access_provisioning", "objective": "objective",
        "applicability": [], "required_inputs": [], "preconditions": [],
        "ordered_steps": [{"step_id": "step_0", "action_intent": "act",
            "tool_name": "tool", "argument_bindings": {}, "preconditions": [],
            "optional": False, "evidence_refs": ["event_0"]}],
        "decision_hints": [], "verification_steps": [], "stop_conditions": [],
        "escalation_hints": [], "source_trace_ids": ["trace_0"],
        "instruction_provenance": {
            "profile": "SSB-INSTRUCTION-PROVENANCE1", "compiler_input_hash": input_hash,
            "instruction_evidence": {"/objective": ["event_0"], "/ordered_steps/0": ["event_0"]},
        },
        "compiler_manifest_ref": manifest_hash,
    })
    usage = TokenUsage(input_tokens=3, output_tokens=5, total_tokens=8)
    cost = TokenCost(
        currency="USD", input_nanos_per_token=2, output_nanos_per_token=4,
        input_nanos=6, output_nanos=20, total_nanos=26,
    )
    return {
        "artifact_profile": "SSB-COMPILED-SKILL1", "compiler_manifest": manifest,
        "compiler_manifest_hash": manifest_hash, "request_envelope_hash": "sha256:" + "d" * 64,
        "skill_ir": skill, "skill_ir_hash": module.hash_skill_ir(skill),
        "rendered_skill": module.render_skill(skill),
        "rendered_skill_hash": module.hash_rendered_skill(skill),
        "raw_request_hash": "sha256:" + "e" * 64,
        "raw_response_hash": "sha256:" + "f" * 64,
        "structured_output_schema_hash": schema_hash, "usage": usage,
        "cost": cost, "attempts": 1,
    }

def _direct_skill(values, mutate):
    raw = module.skill_ir_projection(values["skill_ir"])
    mutate(raw)
    values["skill_ir"] = SkillIR.model_validate(raw)
    values["skill_ir_hash"] = module.hash_skill_ir(values["skill_ir"])
    values["rendered_skill"] = module.render_skill(values["skill_ir"])
    values["rendered_skill_hash"] = module.hash_rendered_skill(values["skill_ir"])

def _assert_direct_artifact_rejected(values):
    try:
        module.CompiledSkillArtifact(**values)
    except ValueError:
        return
    raise AssertionError("invalid direct artifact was accepted")

def assert_direct_artifact_schema_rejected():
    values = _direct_artifact()
    values["structured_output_schema_hash"] = "sha256:" + "0" * 64
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_manifest_rejected():
    values = _direct_artifact()
    _direct_skill(values, lambda raw: raw.__setitem__("compiler_manifest_ref", "sha256:" + "0" * 64))
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_extra_key_rejected():
    values = _direct_artifact()
    _direct_skill(values, lambda raw: raw["instruction_provenance"].__setitem__("extra", "no"))
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_evidence_type_rejected():
    values = _direct_artifact()
    _direct_skill(values, lambda raw: raw["instruction_provenance"].__setitem__("instruction_evidence", []))
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_profile_rejected():
    values = _direct_artifact()
    _direct_skill(values, lambda raw: raw["instruction_provenance"].__setitem__("profile", "SSB-WRONG1"))
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_input_hash_rejected():
    values = _direct_artifact()
    _direct_skill(values, lambda raw: raw["instruction_provenance"].__setitem__("compiler_input_hash", "sha256:" + "0" * 64))
    _assert_direct_artifact_rejected(values)

def assert_direct_artifact_cost_rejected():
    values = _direct_artifact()
    values["cost"] = TokenCost(
        currency="USD", input_nanos_per_token=2, output_nanos_per_token=4,
        input_nanos=7, output_nanos=20, total_nanos=27,
    )
    _assert_direct_artifact_rejected(values)

run()
"""
