from __future__ import annotations

# ruff: noqa: E501
import ast
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from shadowskillbench.models import (
    Message,
    ModelAdapterError,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ScriptedModelClient,
    TokenPricing,
    TokenUsage,
    TransportResponse,
    TransportTransientError,
)
from shadowskillbench.models.openai_compatible import OpenAICompatibleClient, _strict_json
from shadowskillbench.models.protocol import _MAX_I64, _calculate_cost

ADAPTER_SOURCE = (
    Path(__file__).parents[3] / "src" / "shadowskillbench" / "models" / "openai_compatible.py"
)
PROTOCOL_SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "models" / "protocol.py"
SCRIPTED_SOURCE = Path(__file__).parents[3] / "src" / "shadowskillbench" / "models" / "scripted.py"
_TIMEOUT_SECONDS = 15
_HASH = "sha256:" + "a" * 64
_MISSING = object()

_ADAPTER_MUTANT_SETUP = """\
import asyncio
import json
from pydantic import BaseModel
from shadowskillbench.models.protocol import (
    Message, ModelAdapterError, ModelRequest, ProviderCapabilities,
    TransportResponse, TransportTransientError,
)
class Output(BaseModel):
    value: str
def caps(structured=True):
    return ProviderCapabilities(
        provider='p', model='m', model_version='v', supports_system_role=True,
        supports_developer_role=True, supports_seed=True, supports_structured_output=structured,
    )
def request():
    return ModelRequest(
        messages=(Message(role='user', content='x'),), temperature=0.0, seed=None, max_tokens=1,
    )
def response(value='ok', reordered=False):
    payload = {
        'model': 'm',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps({'value': value})}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }
    if reordered:
        payload = {'usage': payload['usage'], 'choices': payload['choices'], 'model': payload['model']}
        return json.dumps(payload, indent=1).encode()
    return json.dumps(payload, separators=(',', ':')).encode()
"""


class Output(BaseModel):
    value: str


class RuntimeSchema(BaseModel):
    value: str

    @classmethod
    def model_json_schema(cls, **_: object) -> dict[str, object]:
        raise RuntimeError("schema-canary")


class ConstructingInvalidSchema(BaseModel):
    value: str

    @classmethod
    def model_validate(cls, _: object, **__: object) -> ConstructingInvalidSchema:
        return BaseModel.model_construct.__func__(cls, value=123)


def _caps(**updates: object) -> ProviderCapabilities:
    return ProviderCapabilities(
        provider="scripted",
        model="model-test",
        model_version="v1",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=True,
    ).model_copy(update=updates)


def _request(**updates: object) -> ModelRequest:
    return ModelRequest(
        messages=(Message(role="user", content="return JSON"),),
        temperature=0.0,
        seed=7,
        max_tokens=33,
    ).model_copy(update=updates)


def _body(
    *,
    model: str = "model-test",
    index: object = 0,
    finish_reason: object = "stop",
    content: object = '{"value":"ok"}',
    usage: object = None,
    refusal: object = _MISSING,
    include_content: bool = True,
) -> bytes:
    message: dict[str, object] = {"role": "assistant"}
    if include_content:
        message["content"] = content
    if refusal is not _MISSING:
        message["refusal"] = refusal
    payload = {
        "model": model,
        "choices": [
            {
                "index": index,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _client(
    script: tuple[object, ...],
    *,
    max_attempts: int = 1,
    pricing: TokenPricing | None = None,
    capabilities: ProviderCapabilities | None = None,
) -> ScriptedModelClient:
    return ScriptedModelClient(
        capabilities=_caps() if capabilities is None else capabilities,
        script=script,  # type: ignore[arg-type]
        max_attempts=max_attempts,
        pricing=pricing,
    )


def _adapter_error(
    client: OpenAICompatibleClient, schema: type[BaseModel] = Output
) -> ModelAdapterError:
    with pytest.raises(ModelAdapterError) as raised:
        asyncio.run(client.structured(_request(), schema))
    return raised.value


def _perfect_forgery(model: type[BaseModel], fields: dict[str, object]) -> BaseModel:
    forged = object.__new__(model)
    object.__setattr__(forged, "__dict__", dict(fields))
    object.__setattr__(forged, "__pydantic_fields_set__", set(fields))
    object.__setattr__(forged, "__pydantic_extra__", None)
    object.__setattr__(forged, "__pydantic_private__", {"_ssb_valid": True})
    return forged


def _response_kwargs(output: object) -> dict[str, object]:
    return {
        "capabilities": _caps(),
        "output": output,
        "raw_request_hash": _HASH,
        "raw_response_hash": _HASH,
        "structured_output_schema_hash": _HASH,
        "usage": TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        "cost": None,
        "attempts": 1,
    }


_SAME_INSTANCE_FALLBACK_MUTANT_BODY = """\
from pydantic import BaseModel, ConfigDict, field_validator
class SameInstanceFallback(BaseModel):
    model_config = ConfigDict(revalidate_instances='never')
    value: tuple[str, ...]
    @field_validator('value', mode='before')
    @classmethod
    def wire_lists_only(cls, value):
        if type(value) is not list:
            raise ValueError('value must be a wire list')
        return tuple(value)
output = SameInstanceFallback(value=['wire'])
kwargs = {
    'capabilities': module.ProviderCapabilities(
        provider='p', model='m', model_version='v', supports_system_role=True,
        supports_developer_role=True, supports_seed=True, supports_structured_output=True,
    ),
    'raw_request_hash': 'sha256:' + 'a' * 64,
    'raw_response_hash': 'sha256:' + 'b' * 64,
    'structured_output_schema_hash': 'sha256:' + 'c' * 64,
    'usage': module.TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    'cost': None,
    'attempts': 1,
}
try:
    module.ModelResponse[SameInstanceFallback](output=output, **kwargs)
except ValueError:
    pass
else:
    raise AssertionError('same-instance fallback accepted')
"""


def _run_mutant(
    tmp_path: Path, source_path: Path, module_name: str, needle: str, replacement: str, body: str
) -> subprocess.CompletedProcess[str]:
    source = source_path.read_text(encoding="utf-8")
    assert needle in source
    mutant = tmp_path / f"{module_name.rsplit('.', 1)[-1]}_mutant.py"
    check = tmp_path / "check.py"
    mutant.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
    indented = "\n".join(f"    {line}" if line else line for line in body.splitlines())
    check.write_text(
        "import importlib.util\nimport sys\n"
        f"spec = importlib.util.spec_from_file_location({module_name!r}, sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec.loader is not None\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        f"try:\n{indented}\n"
        "except AssertionError:\n    raise\n"
        "except Exception as error:\n    raise AssertionError('mutant behavior rejected') from error\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(check), str(mutant)],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )


def _assert_killed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 1, result.stderr
    assert "AssertionError" in result.stderr, result.stderr


@pytest.mark.mutation
def test_schema_validation_stays_strict_and_forbids_extra_output_fields(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        PROTOCOL_SOURCE,
        "shadowskillbench.models.protocol_mutant",
        'raw, strict=True, extra="forbid", from_attributes=False',
        'raw, strict=False, extra="ignore", from_attributes=False',
        "from pydantic import BaseModel\n"
        "class Output(BaseModel):\n    value: str\n"
        "try:\n    module._validate_schema_output(Output, {'value': '7', 'extra': True})\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('extra output accepted')",
    )
    _assert_killed(result)


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "if output is value or type(output) is not schema:",
            "if type(output) is not schema:",
        ),
        ("return _validate_schema_output(schema, raw)", "return value"),
    ],
    ids=("distinct-instance", "mapping-fallback-gate"),
)
def test_caller_revalidation_mutants_are_killed(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    result = _run_mutant(
        tmp_path,
        PROTOCOL_SOURCE,
        "shadowskillbench.models.protocol_mutant",
        needle,
        replacement,
        _SAME_INSTANCE_FALLBACK_MUTANT_BODY,
    )
    _assert_killed(result)


@pytest.mark.mutation
def test_parser_post_validation_revalidation_mutant_is_killed(tmp_path: Path) -> None:
    result = _run_mutant(
        tmp_path,
        ADAPTER_SOURCE,
        "shadowskillbench.models.adapter_mutant",
        "return cast(T, _revalidate_caller_output(output, schema)), usage",
        "return cast(T, output), usage",
        _ADAPTER_MUTANT_SETUP + "from pydantic import ConfigDict, field_validator\n"
        "class SameInstanceFallback(BaseModel):\n"
        "    model_config = ConfigDict(revalidate_instances='never')\n"
        "    value: tuple[str, ...]\n"
        "    @field_validator('value', mode='before')\n"
        "    @classmethod\n"
        "    def wire_lists_only(cls, value):\n"
        "        if type(value) is not list:\n"
        "            raise ValueError('value must be a wire list')\n"
        "        return tuple(value)\n"
        "async def unused(_):\n"
        "    raise AssertionError('transport is unused')\n"
        "payload = {\n"
        "    'model': 'm',\n"
        "    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps({'value': ['wire']})}, 'finish_reason': 'stop'}],\n"
        "    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},\n"
        "}\n"
        "client = module.OpenAICompatibleClient(capabilities=caps(), transport=unused, max_attempts=1)\n"
        "try:\n"
        "    client._parse_success(json.dumps(payload, separators=(',', ':')).encode(), SameInstanceFallback)\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('parser post-validation fallback was bypassed')\n",
    )
    _assert_killed(result)


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("source_path", "module_name", "needle", "replacement", "body"),
    [
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            'response_hash = "sha256:" + hashlib.sha256(returned.body).hexdigest()',
            'response_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(_strict_json(returned.body))).hexdigest()',
            _ADAPTER_MUTANT_SETUP
            + (
                "async def compact(_):\n"
                "    return TransportResponse(status_code=200, body=response())\n"
                "async def reordered(_):\n"
                "    return TransportResponse(status_code=200, body=response(reordered=True))\n"
                "first = asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=compact, max_attempts=1).structured(request(), Output))\n"
                "second = asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=reordered, max_attempts=1).structured(request(), Output))\n"
                "assert first.output == second.output == Output(value='ok')\n"
                "assert first.raw_response_hash != second.raw_response_hash\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "if transport_code is None:\n                    continue",
            "if transport_code is None:\n                    pass",
            _ADAPTER_MUTANT_SETUP
            + (
                "steps = [TransportTransientError(), TransportResponse(status_code=200, body=response())]\n"
                "sent = []\n"
                "async def transport(body):\n"
                "    sent.append(body)\n"
                "    step = steps.pop(0)\n"
                "    if isinstance(step, BaseException):\n"
                "        raise step\n"
                "    return step\n"
                "result = asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=transport, max_attempts=2).structured(request(), Output))\n"
                "assert result.attempts == 2\n"
                "assert len(sent) == 2\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "if not transient:",
            "if False:",
            _ADAPTER_MUTANT_SETUP
            + (
                "steps = [RuntimeError('terminal'), TransportResponse(status_code=200, body=response())]\n"
                "sent = []\n"
                "async def transport(body):\n"
                "    sent.append(body)\n"
                "    step = steps.pop(0)\n"
                "    if isinstance(step, BaseException):\n"
                "        raise step\n"
                "    return step\n"
                "try:\n"
                "    asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=transport, max_attempts=2).structured(request(), Output))\n"
                "except ModelAdapterError as error:\n"
                "    assert error.code == 'MODEL_PROVIDER_TERMINAL'\n"
                "    assert error.attempts == 1\n"
                "else:\n"
                "    raise AssertionError('terminal provider failure retried')\n"
                "assert len(sent) == 1\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "if parsed_result is None:",
            "if parsed_result is None:\n                continue\n            if False:",
            _ADAPTER_MUTANT_SETUP
            + (
                "steps = [TransportResponse(status_code=200, body=b'not-json'), TransportResponse(status_code=200, body=response())]\n"
                "sent = []\n"
                "async def transport(body):\n"
                "    sent.append(body)\n"
                "    return steps.pop(0)\n"
                "try:\n"
                "    asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=transport, max_attempts=2).structured(request(), Output))\n"
                "except ModelAdapterError as error:\n"
                "    assert error.code == 'MODEL_OUTPUT_INVALID'\n"
                "    assert error.attempts == 1\n"
                "else:\n"
                "    raise AssertionError('invalid output retried')\n"
                "assert len(sent) == 1\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "not caps.supports_structured_output",
            "False",
            _ADAPTER_MUTANT_SETUP
            + (
                "sent = []\n"
                "async def transport(body):\n"
                "    sent.append(body)\n"
                "    return TransportResponse(status_code=200, body=response())\n"
                "try:\n"
                "    asyncio.run(module.OpenAICompatibleClient(capabilities=caps(False), transport=transport, max_attempts=1).structured(request(), Output))\n"
                "except ModelAdapterError as error:\n"
                "    assert error.code == 'CONFIGURATION_ERROR'\n"
                "    assert error.attempts == 0\n"
                "else:\n"
                "    raise AssertionError('unsupported capability bypassed preflight')\n"
                "assert sent == []\n"
            ),
        ),
        (
            PROTOCOL_SOURCE,
            "shadowskillbench.models.protocol_mutant",
            "if self.total_tokens != self.input_tokens + self.output_tokens:",
            "if False:",
            (
                "try:\n"
                "    module.TokenUsage(input_tokens=1, output_tokens=1, total_tokens=3)\n"
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('inconsistent usage admitted')\n"
            ),
        ),
        (
            PROTOCOL_SOURCE,
            "shadowskillbench.models.protocol_mutant",
            "total = input_nanos + output_nanos",
            "total = input_nanos - output_nanos",
            (
                "usage = module.TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5)\n"
                "pricing = module.TokenPricing(currency='USD', input_nanos_per_token=7, output_nanos_per_token=11)\n"
                "cost = module._calculate_cost(usage, pricing)\n"
                "assert (cost.input_nanos, cost.output_nanos, cost.total_nanos) == (14, 33, 47)\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "if type(key) is not str or key in result:",
            "if type(key) is not str:",
            (
                "try:\n"
                '    module._strict_json(b\'{\\"x\\":1,\\"x\\":2}\')\n'
                "except ValueError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('duplicate JSON key admitted')\n"
            ),
        ),
        (
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            'transport_code = "MODEL_PROVIDER_TERMINAL"',
            'raise self._error("MODEL_PROVIDER_TERMINAL", attempts, True, request_hash, None)',
            _ADAPTER_MUTANT_SETUP
            + (
                "async def transport(_):\n"
                "    raise RuntimeError('SECRET_MUTANT_CANARY')\n"
                "try:\n"
                "    asyncio.run(module.OpenAICompatibleClient(capabilities=caps(), transport=transport, max_attempts=1).structured(request(), Output))\n"
                "except ModelAdapterError as error:\n"
                "    assert BaseException.__getattribute__(error, '__cause__') is None\n"
                "    assert BaseException.__getattribute__(error, '__context__') is None\n"
                "    assert 'SECRET_MUTANT_CANARY' not in repr(error)\n"
                "else:\n"
                "    raise AssertionError('terminal error was not surfaced')\n"
            ),
        ),
    ],
    ids=(
        "raw-response-hash",
        "transient-retry",
        "terminal-retry",
        "output-retry",
        "capability-preflight",
        "usage-sum",
        "cost-arithmetic",
        "duplicate-key",
        "secret-context",
    ),
)
def test_executable_adapter_contract_mutants_are_killed(
    tmp_path: Path,
    source_path: Path,
    module_name: str,
    needle: str,
    replacement: str,
    body: str,
) -> None:
    _assert_killed(_run_mutant(tmp_path, source_path, module_name, needle, replacement, body))


@pytest.mark.mutation
def test_raw_response_hash_is_not_canonical_or_semantic_hash() -> None:
    first_body = _body()
    reordered = json.dumps(
        {
            "usage": {"total_tokens": 5, "completion_tokens": 3, "prompt_tokens": 2},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"value":"ok"}', "role": "assistant"},
                    "index": 0,
                }
            ],
            "model": "model-test",
        },
        indent=2,
    ).encode("utf-8")
    one = asyncio.run(
        _client((TransportResponse(status_code=200, body=first_body),)).structured(
            _request(), Output
        )
    )
    two = asyncio.run(
        _client((TransportResponse(status_code=200, body=reordered),)).structured(
            _request(), Output
        )
    )
    assert one.output == two.output == Output(value="ok")
    assert one.raw_response_hash == "sha256:" + hashlib.sha256(first_body).hexdigest()
    assert two.raw_response_hash == "sha256:" + hashlib.sha256(reordered).hexdigest()
    assert one.raw_response_hash != two.raw_response_hash


@pytest.mark.mutation
def test_capability_failure_precedes_schema_but_schema_runtime_is_not_configuration() -> None:
    unsupported = _client((), capabilities=_caps(supports_seed=False))
    configuration = _adapter_error(unsupported, RuntimeSchema)
    assert configuration.code == "CONFIGURATION_ERROR"
    assert configuration.attempts == 0 and configuration.raw_request_hash is None
    schema = _adapter_error(_client(()), RuntimeSchema)
    assert schema.code == "SCHEMA_ERROR"
    assert schema.attempts == 0 and schema.raw_request_hash is None


@pytest.mark.mutation
def test_transient_terminal_output_and_cost_paths_have_exact_retry_behavior() -> None:
    class TransientSubclass(TransportTransientError):
        pass

    subclassed = _client(("transient",), max_attempts=2)
    subclassed._script = (TransientSubclass(),)  # type: ignore[assignment]
    terminal = _adapter_error(subclassed)
    assert terminal.code == "MODEL_PROVIDER_TERMINAL" and terminal.attempts == 1
    transient = _client((TransportResponse(status_code=503, body=b"first"),), max_attempts=1)
    transient_error = _adapter_error(transient)
    assert transient_error.code == "MODEL_PROVIDER_TRANSIENT"
    assert transient_error.raw_response_hash == "sha256:" + hashlib.sha256(b"first").hexdigest()
    invalid = _client(
        (TransportResponse(status_code=200, body=_body(content="not json")),), max_attempts=2
    )
    invalid_error = _adapter_error(invalid)
    assert invalid_error.code == "MODEL_OUTPUT_INVALID" and invalid_error.attempts == 1
    assert len(invalid.recorded_request_bodies) == 1
    pricing = TokenPricing(currency="USD", input_nanos_per_token=_MAX_I64, output_nanos_per_token=1)
    costly = _client(
        (TransportResponse(status_code=200, body=_body()),), max_attempts=2, pricing=pricing
    )
    costly_error = _adapter_error(costly)
    assert costly_error.code == "CONFIGURATION_ERROR" and costly_error.attempts == 1
    assert len(costly.recorded_request_bodies) == 1


@pytest.mark.mutation
def test_status_bounds_choice_index_duplicate_nonfinite_and_cost_arithmetic_are_exact() -> None:
    for status in (99, 600, True):
        with pytest.raises(ValueError):
            TransportResponse(status_code=status, body=b"")
    for index in (True, 1, -1):
        error = _adapter_error(
            _client((TransportResponse(status_code=200, body=_body(index=index)),))
        )
        assert error.code == "MODEL_OUTPUT_INVALID" and error.before_meaningful_behavior is False
    for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
        with pytest.raises(ValueError, match="invalid JSON"):
            _strict_json(payload)
    cost = _calculate_cost(
        TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        TokenPricing(currency="USD", input_nanos_per_token=7, output_nanos_per_token=11),
    )
    assert (cost.input_nanos, cost.output_nanos, cost.total_nanos) == (14, 33, 47)
    with pytest.raises(OverflowError, match="cost overflow"):
        _calculate_cost(
            TokenUsage(input_tokens=2, output_tokens=0, total_tokens=2),
            TokenPricing(currency="USD", input_nanos_per_token=_MAX_I64, output_nanos_per_token=0),
        )


@pytest.mark.mutation
def test_final_transient_response_hash_is_from_the_final_attempt_not_a_prior_attempt() -> None:
    first = b"first-transient"
    final = b"final-transient"
    client = _client(
        (
            TransportResponse(status_code=503, body=first),
            TransportResponse(status_code=429, body=final),
        ),
        max_attempts=2,
    )
    error = _adapter_error(client)
    assert error.code == "MODEL_PROVIDER_TRANSIENT" and error.attempts == 2
    assert error.raw_response_hash == "sha256:" + hashlib.sha256(final).hexdigest()
    assert error.raw_response_hash != "sha256:" + hashlib.sha256(first).hexdigest()
    assert client.recorded_request_bodies == (client.recorded_request_bodies[0],) * 2


@pytest.mark.mutation
@pytest.mark.parametrize(
    ("body", "expected_before"),
    [
        (b"{}", True),
        (_body(model="other"), False),
        (_body(finish_reason="length"), False),
        (_body(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 9}), False),
        (_body(content="not-json"), False),
        (_body(content='{"value":7}'), False),
        (_body(include_content=False, refusal="declined"), False),
        (_body(content=7), True),
        (_body(include_content=False, refusal=7), True),
    ],
)
def test_meaningful_behavior_flag_tracks_extractable_nonblank_content(
    body: bytes, expected_before: bool
) -> None:
    error = _adapter_error(_client((TransportResponse(status_code=200, body=body),)))
    assert error.code == "MODEL_OUTPUT_INVALID"
    assert error.before_meaningful_behavior is expected_before


@pytest.mark.mutation
def test_error_graph_has_no_secret_cause_or_context_even_via_base_exception() -> None:
    async def secret_transport(_: bytes) -> TransportResponse:
        raise RuntimeError("SECRET_CANARY")

    from shadowskillbench.models.openai_compatible import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        capabilities=_caps(), transport=secret_transport, max_attempts=1
    )
    error = _adapter_error(client)
    assert error.code == "MODEL_PROVIDER_TERMINAL"
    assert "SECRET_CANARY" not in repr(error)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None


@pytest.mark.mutation
def test_model_ingress_copy_subclass_container_and_script_markers_are_rejected() -> None:
    class RequestSubclass(ModelRequest):
        pass

    class CustomList(list[object]):
        pass

    class Marker(str):
        pass

    with pytest.raises(ValueError):
        ModelRequest(
            messages=CustomList([Message(role="user", content="x")]),
            temperature=0.0,
            seed=None,
            max_tokens=1,
        )
    with pytest.raises(ValueError):
        RequestSubclass(**_request().model_dump())
    forged = _request()
    object.__getattribute__(forged, "__dict__")["unexpected"] = "value"
    client = _client(())
    with pytest.raises(ValueError):
        client._preflight(forged, Output)
    copied = _request().model_copy()
    baseline = _request()
    assert copied == baseline and copied is not baseline
    with pytest.raises(ValueError):
        _client((Marker("transient"),))


@pytest.mark.mutation
def test_perfect_caller_set_attestation_does_not_admit_a_forged_request() -> None:
    fields = dict(object.__getattribute__(_request(), "__dict__"))
    forged = _perfect_forgery(ModelRequest, fields)

    with pytest.raises(ValueError):
        _client(())._preflight(forged, Output)  # type: ignore[arg-type]


@pytest.mark.mutation
def test_perfect_caller_set_attestation_does_not_admit_forged_capabilities() -> None:
    fields = dict(object.__getattribute__(_caps(), "__dict__"))
    forged = _perfect_forgery(ProviderCapabilities, fields)

    async def unused_transport(_: bytes) -> TransportResponse:
        raise AssertionError("forged capabilities must fail before transport")

    with pytest.raises(ValueError):
        OpenAICompatibleClient(  # type: ignore[arg-type]
            capabilities=forged, transport=unused_transport, max_attempts=1
        )


@pytest.mark.mutation
def test_perfect_caller_set_attestation_does_not_admit_a_forged_transport_response() -> None:
    fields = dict(
        object.__getattribute__(TransportResponse(status_code=200, body=_body()), "__dict__")
    )
    forged = _perfect_forgery(TransportResponse, fields)

    async def forged_transport(_: bytes) -> TransportResponse:
        return forged  # type: ignore[return-value]

    error = _adapter_error(
        OpenAICompatibleClient(capabilities=_caps(), transport=forged_transport, max_attempts=1)
    )
    assert error.code == "MODEL_PROVIDER_TERMINAL"
    assert error.raw_response_hash is None


@pytest.mark.mutation
def test_type_adapter_python_ingress_rejects_forged_and_subclassed_requests() -> None:
    class RequestSubclass(ModelRequest):
        pass

    adapter = TypeAdapter(ModelRequest)
    forged = _perfect_forgery(ModelRequest, dict(object.__getattribute__(_request(), "__dict__")))
    subclass = _perfect_forgery(
        RequestSubclass, dict(object.__getattribute__(_request(), "__dict__"))
    )

    with pytest.raises(ValueError):
        adapter.validate_python(forged)
    with pytest.raises(ValueError):
        adapter.validate_python(subclass)


@pytest.mark.mutation
def test_model_response_rejects_output_subclasses_and_invalid_forged_outputs() -> None:
    class OutputSubclass(Output):
        pass

    forged = object.__new__(Output)
    object.__setattr__(forged, "__dict__", {"value": 7})
    object.__setattr__(forged, "__pydantic_fields_set__", {"value"})
    object.__setattr__(forged, "__pydantic_extra__", None)
    object.__setattr__(forged, "__pydantic_private__", None)

    with pytest.raises(ValueError):
        ModelResponse[Output](**_response_kwargs(OutputSubclass(value="ok")))
    with pytest.raises(ValueError):
        ModelResponse[Output](**_response_kwargs(forged))


@pytest.mark.mutation
def test_adapter_rejects_a_schema_that_constructs_invalid_exact_type_output() -> None:
    body = _body(content='{"value":"wire-value"}')
    client = _client((TransportResponse(status_code=200, body=body),), max_attempts=2)

    error = _adapter_error(client, ConstructingInvalidSchema)

    assert error.code == "MODEL_OUTPUT_INVALID"
    assert error.attempts == 1
    assert error.before_meaningful_behavior is False
    assert error.raw_response_hash == "sha256:" + hashlib.sha256(body).hexdigest()
    assert len(client.recorded_request_bodies) == 1


@pytest.mark.mutation
def test_model_response_revalidates_invalid_exact_type_schema_output() -> None:
    invalid = ConstructingInvalidSchema.model_validate({"value": "wire-value"})
    assert type(invalid) is ConstructingInvalidSchema
    assert type(invalid.value) is int and invalid.value == 123

    with pytest.raises(ValueError):
        ModelResponse[ConstructingInvalidSchema](**_response_kwargs(invalid))

    valid = ModelResponse[Output](**_response_kwargs(Output(value="valid-caller-output")))
    assert type(valid.output) is Output
    assert valid.output.value == "valid-caller-output"


@pytest.mark.mutation
def test_model_adapter_error_requires_exact_code_and_a_request_hash_after_attempts() -> None:
    class CodeSubclass(str):
        pass

    with pytest.raises(ValueError):
        ModelAdapterError(
            code=CodeSubclass("MODEL_PROVIDER_TERMINAL"),
            attempts=1,
            before_meaningful_behavior=True,
            raw_request_hash=_HASH,
            raw_response_hash=None,
        )
    with pytest.raises(ValueError):
        ModelAdapterError(
            code="MODEL_PROVIDER_TERMINAL",
            attempts=1,
            before_meaningful_behavior=True,
            raw_request_hash=None,
            raw_response_hash=None,
        )


@pytest.mark.mutation
def test_model_adapter_error_rejects_post_construction_mutation_and_secret_fields() -> None:
    error = ModelAdapterError(
        code="MODEL_PROVIDER_TERMINAL",
        attempts=1,
        before_meaningful_behavior=True,
        raw_request_hash=_HASH,
        raw_response_hash=None,
    )

    with pytest.raises((AttributeError, TypeError)):
        error.code = "MODEL_OUTPUT_INVALID"
    with pytest.raises((AttributeError, TypeError)):
        error.SECRET_CANARY = "secret"  # type: ignore[attr-defined]


@pytest.mark.mutation
@pytest.mark.parametrize("final_kind", ("exception", "wrong_response"))
def test_final_transport_failure_after_a_prior_transient_has_no_response_hash(
    final_kind: str,
) -> None:
    attempts = 0

    async def transport(_: bytes) -> TransportResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return TransportResponse(status_code=503, body=b"prior-transient")
        if final_kind == "exception":
            raise RuntimeError("terminal-after-transient")
        return object()  # type: ignore[return-value]

    error = _adapter_error(
        OpenAICompatibleClient(capabilities=_caps(), transport=transport, max_attempts=2)
    )
    assert error.code == "MODEL_PROVIDER_TERMINAL"
    assert error.attempts == 2
    assert error.raw_request_hash is not None
    assert error.raw_response_hash is None


@pytest.mark.mutation
def test_adapter_sources_have_no_ambient_or_compiler_import_surface(tmp_path: Path) -> None:
    forbidden_modules = {
        "openai",
        "anthropic",
        "http",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "os",
        "pathlib",
        "subprocess",
        "random",
        "time",
        "datetime",
        "logging",
        "compiler",
    }
    forbidden_calls = {
        "sleep",
        "time",
        "monotonic",
        "perf_counter",
        "open",
        "read_text",
        "write_text",
        "unlink",
        "remove",
        "rmdir",
        "mkdir",
        "makedirs",
        "getenv",
        "putenv",
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "get",
        "post",
        "request",
        "urlopen",
        "connect",
        "send",
        "recv",
        "random",
        "randint",
        "choice",
        "basicConfig",
        "getLogger",
    }
    for source_path in (ADAPTER_SOURCE, PROTOCOL_SOURCE, SCRIPTED_SOURCE):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        forbidden_imports = {
            module
            for module in imported
            if module.split(".", 1)[0] in forbidden_modules
            or module.startswith("shadowskillbench.compiler")
            or module.startswith("shadowskillbench.skills")
        }
        skill_ir_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "SkillIR"
        }
        assert not forbidden_imports, (source_path, forbidden_imports)
        assert not skill_ir_imports, (source_path, skill_ir_imports)
        assert not (called_names & forbidden_calls), (
            source_path,
            called_names & forbidden_calls,
        )
    for forbidden_import in ("os", "pathlib", "socket", "logging"):
        result = _run_mutant(
            tmp_path,
            ADAPTER_SOURCE,
            "shadowskillbench.models.adapter_mutant",
            "import json",
            f"import json\nimport {forbidden_import}",
            "from pathlib import Path\n"
            f"assert 'import {forbidden_import}' not in Path(sys.argv[1]).read_text()",
        )
        _assert_killed(result)
