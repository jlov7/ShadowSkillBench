from __future__ import annotations

# ruff: noqa: E501
import subprocess
import sys
from pathlib import Path

import pytest

MODELS = Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "models.py"
RENDER = Path(__file__).parents[3] / "src" / "shadowskillbench" / "skills" / "render.py"
_TIMEOUT_SECONDS = 15


def _run_mutant(
    tmp_path: Path, source_path: Path, module_name: str, needle: str, replacement: str, body: str
) -> subprocess.CompletedProcess[str]:
    source = source_path.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / f"{module_name.rsplit('.', 1)[-1]}_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    check.write_text(_load(module_name) + _wrapped(body), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(check), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )


def _load(module_name: str) -> str:
    return f"""\\
import importlib.util
import sys
spec = importlib.util.spec_from_file_location({module_name!r}, sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""


def _wrapped(body: str) -> str:
    indented = "\n".join(f"    {line}" if line else line for line in body.splitlines())
    return (
        "try:\n"
        f"{indented}\n"
        "except AssertionError:\n"
        "    raise\n"
        "except Exception as error:\n"
        "    raise AssertionError('mutant behavior rejected') from error\n"
    )


_RAW = """\\
def raw():
    return {
        'skill_id': 'skill_mutant', 'schema_version': '1.0', 'domain': 'access_provisioning',
        'objective': 'Review request.', 'applicability': [], 'required_inputs': [], 'preconditions': [],
        'ordered_steps': [{'step_id': 'review', 'action_intent': 'Review request.', 'tool_name': 'inspect',
            'argument_bindings': {}, 'preconditions': [], 'optional': False, 'evidence_refs': []}],
        'decision_hints': [], 'verification_steps': [], 'stop_conditions': [], 'escalation_hints': [],
        'source_trace_ids': ['trace_mutant'], 'instruction_provenance': {},
        'compiler_manifest_ref': 'sha256:' + 'a' * 64,
    }
"""


def _models_case(needle: str, replacement: str, body: str, tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path, MODELS, "shadowskillbench.skills.models_mutant", needle, replacement, _RAW + body
    )
    assert result.returncode == 1, result.stderr
    assert "AssertionError" in result.stderr, result.stderr


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "body"),
    [
        (
            "if any(type(key) is not str for key in raw):",
            "if False:",
            "class S(str):\n    def __eq__(self, other):\n        raise AssertionError('hostile key equality invoked')\n    __hash__ = str.__hash__\n"
            "value = raw(); value[S('skill_id')] = value.pop('skill_id')\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('subclassed key accepted')\n",
        ),
        (
            'if type(value) is not str or value != "1.0":',
            "if False:",
            "value = raw(); value['schema_version'] = '2.0'\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('weak version accepted')\n",
        ),
        (
            "if type(value) is not str or len(value) > _MAX_IDENTIFIER_LENGTH or value not in _DOMAINS:",
            "if False:",
            "value = raw(); value['domain'] = 'other'\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('weak domain accepted')\n",
        ),
        (
            "if len({step.step_id for step in self.ordered_steps}) != len(self.ordered_steps):",
            "if False:",
            "value = raw(); value['ordered_steps'].append({'step_id': 'review', 'action_intent': 'Different.', 'tool_name': 'other', 'argument_bindings': {}, 'preconditions': [], 'optional': False, 'evidence_refs': []})\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('duplicate steps accepted')\n",
        ),
        (
            "key in _FORBIDDEN_KEYS",
            "False",
            "value = raw(); value['instruction_provenance'] = {'worker_policy_id': 'x'}\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('treatment key accepted')\n",
        ),
        (
            "or text in _FORBIDDEN_VALUES",
            "or False",
            "value = raw(); value['ordered_steps'][0]['tool_name'] = 'access_compliant_v1'\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('treatment identifier accepted')\n",
        ),
        (
            "or pattern.fullmatch(text) is None",
            "or False",
            "value = raw(); value['skill_id'] = 'skill invalid'\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('invalid identifier grammar accepted')\n",
        ),
        (
            'return tuple(_text(item, field=f"{field} item") for item in raw)',
            'return tuple(sorted(_text(item, field=f"{field} item") for item in raw))',
            "left = raw(); left['applicability'] = ['first', 'second']\n"
            "right = raw(); right['applicability'] = ['second', 'first']\n"
            "assert module.hash_skill_ir(left) != module.hash_skill_ir(right)\n",
        ),
        (
            "if any(type(key) is not str for key in copied):",
            "if False:",
            "class S(str):\n    def __eq__(self, other):\n        raise AssertionError('hostile update equality invoked')\n    __hash__ = str.__hash__\n"
            "parsed = module.parse_skill_ir(raw())\n"
            "try:\n    parsed.model_copy(update={S('objective'): 'Changed.'})\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('subclassed update key accepted')\n",
        ),
        (
            'raise ValueError("legacy JSON/string ingress is forbidden")',
            "return cls.model_construct()",
            "try:\n    module.SkillIR.parse_raw('{}')\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('legacy JSON ingress accepted')\n",
        ),
        (
            'raise ValueError("legacy filesystem ingress is forbidden")',
            "return cls.model_construct()",
            "try:\n    module.SkillIR.parse_file('valid.json')\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('legacy filesystem ingress accepted')\n",
        ),
        (
            "return self.model_copy(update=update, deep=deep)",
            "return self",
            "parsed = module.parse_skill_ir(raw())\nassert parsed.copy() is not parsed\n",
        ),
        (
            "elif type(self) is SkillIR:\n            _preflight_model_skill(cast(SkillIR, self))\n            raw = _model_to_raw(cast(SkillIR, self))",
            "elif type(self) is SkillIR:\n            raw = _model_to_raw(cast(SkillIR, self))",
            "parsed = module.parse_skill_ir(raw()); shared = {}\n"
            "object.__getattribute__(parsed, '__dict__')['instruction_provenance'] = shared\n"
            "object.__getattribute__(parsed.ordered_steps[0], '__dict__')['argument_bindings'] = shared\n"
            "try:\n    parsed.model_copy()\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('forged copy alias accepted')\n",
        ),
        (
            '    _scan_json_root(raw["instruction_provenance"], field="instruction_provenance", state=state)\n    return None',
            "    return None",
            "parsed = module.parse_skill_ir(raw()); shared = {}\n"
            "object.__getattribute__(parsed, '__dict__')['instruction_provenance'] = shared\n"
            "object.__getattribute__(parsed.ordered_steps[0], '__dict__')['argument_bindings'] = shared\n"
            "try:\n    module.parse_skill_ir(parsed)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('forged cross-root alias accepted')\n",
        ),
        (
            "return sha256_ref(skill_ir_projection(value))",
            "return sha256_ref({})",
            "assert module.hash_skill_ir(raw()) != module.hash_skill_ir({**raw(), 'objective': 'Changed.'})\n",
        ),
        (
            "if len(children) > _MAX_JSON_VALUES - state.json_values - len(stack):",
            "if False:",
            "value = raw(); value['instruction_provenance'] = {'values': [0] * 60000}\n"
            "value['ordered_steps'][0]['argument_bindings'] = {'values': [0] * 60000}\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('aggregate JSON budget accepted')\n",
        ),
        (
            "if container_id in self.seen:",
            "if False:",
            "value = raw(); shared = {}; value['instruction_provenance'] = shared\n"
            "value['ordered_steps'][0]['argument_bindings'] = shared\n"
            "try:\n    module.parse_skill_ir(value)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('cross-root alias accepted')\n",
        ),
        (
            'projection = cast(JsonObject, _owned_json_object(raw, field="skill_ir_projection"))',
            'projection = {"skill_id": raw["skill_id"]}',
            "projection = module.skill_ir_projection(raw())\n"
            "assert tuple(projection) == ('skill_id', 'schema_version', 'domain', 'objective', 'applicability', 'required_inputs', 'preconditions', 'ordered_steps', 'decision_hints', 'verification_steps', 'stop_conditions', 'escalation_hints', 'source_trace_ids', 'instruction_provenance', 'compiler_manifest_ref')\n",
        ),
        (
            "return _step_to_raw(cast(OrderedStep, value))",
            "return _model_data(value, expected=cls, fields=cls._fields)",
            "from pydantic import TypeAdapter\n"
            "parsed = module.parse_skill_ir(raw())\n"
            "result = TypeAdapter(module.OrderedStep).validate_python(parsed.ordered_steps[0])\n"
            "assert result is not parsed.ordered_steps[0] and result == parsed.ordered_steps[0]\n",
        ),
    ],
)
def test_mutation_kills_skill_ir_contract_bypasses(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> None:
    _models_case(needle, replacement, body, tmp_path)


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement", "body"),
    [
        (
            'return "\\n".join(lines) + "\\n"',
            'return "\\n".join(lines)',
            _RAW + "assert module.render_skill(raw()).endswith('\\n')\n",
        ),
        (
            'f"    {arguments}",',
            "arguments,",
            _RAW + "assert '\\n    {\"' in module.render_skill(raw())\n",
        ),
        (
            'for character in ("`", "*", "_", "[", "]", "<", ">", "#", "|"):',
            "for character in ():",
            _RAW
            + "value = raw(); value['objective'] = 'Escaped *item*'\n"
            + "assert 'Escaped \\\\*item\\\\*' in module.render_skill(value)\n",
        ),
        (
            'f"# {_escape_markdown(parsed.skill_id)}",',
            'f"# {parsed.skill_id}",',
            _RAW + "assert '# skill\\\\_mutant' in module.render_skill(raw())\n",
        ),
        (
            'f"Domain: {_escape_markdown(parsed.domain)}",',
            'f"Domain: {parsed.domain}",',
            _RAW + "assert 'Domain: access\\\\_provisioning' in module.render_skill(raw())\n",
        ),
        (
            '("Verification", parsed.verification_steps),',
            '("Verification", parsed.stop_conditions),',
            _RAW
            + "value = raw(); value['verification_steps'] = ['verify']; value['stop_conditions'] = ['stop']\n"
            + "rendered = module.render_skill(value)\nassert '## Verification\\n\\n- verify' in rendered\n",
        ),
        (
            'return "\\n".join(lines) + "\\n"',
            'return "\\n".join(lines + list(parsed.source_trace_ids)) + "\\n"',
            _RAW + "assert 'trace_mutant' not in module.render_skill(raw())\n",
        ),
    ],
)
def test_mutation_kills_renderer_contract_bypasses(
    tmp_path: Path, needle: str, replacement: str, body: str
) -> None:
    result = _run_mutant(
        tmp_path,
        RENDER,
        "shadowskillbench.skills.render_mutant",
        needle,
        replacement,
        body,
    )
    assert result.returncode == 1, result.stderr
    assert "AssertionError" in result.stderr, result.stderr


@pytest.mark.mutation
def test_mutation_kills_forbidden_ambient_import(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        MODELS,
        "shadowskillbench.skills.models_mutant",
        "import math",
        "import math\nimport pathlib",
        "from pathlib import Path\nassert 'import pathlib' not in Path(sys.argv[1]).read_text()\n",
    )
    assert result.returncode == 1, result.stderr
    assert "AssertionError" in result.stderr, result.stderr
