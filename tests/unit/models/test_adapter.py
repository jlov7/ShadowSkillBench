from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from shadowskillbench.models import (
    AsyncTransport,
    Message,
    ModelAdapterError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleClient,
    ProviderCapabilities,
    ScriptedModelClient,
    TokenCost,
    TokenPricing,
    TokenUsage,
    TransportResponse,
)
from shadowskillbench.skills import SkillIR


class Output(BaseModel):
    value: str


class OtherOutput(BaseModel):
    value: str


def _caps(**changes: object) -> ProviderCapabilities:
    values: dict[str, object] = dict(
        provider="scripted",
        model="model-test",
        model_version="v1",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=True,
    )
    values.update(changes)
    return ProviderCapabilities(**values)


def _request(**changes: object) -> ModelRequest:
    values: dict[str, object] = dict(
        messages=(Message(role="user", content="return JSON"),),
        temperature=0.0,
        seed=7,
        max_tokens=33,
    )
    values.update(changes)
    return ModelRequest(**values)


def _body(
    *,
    model: object = "model-test",
    choices: object | None = None,
    usage: object | None = None,
    extra: Mapping[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "model": model,
        "choices": choices
        if choices is not None
        else [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"value":"ok"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _skill_ir_raw(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "skill_id": "skill_adapter_compatibility",
        "schema_version": "1.0",
        "domain": "access_provisioning",
        "objective": "Review the request.",
        "applicability": ["Use for a visible request."],
        "required_inputs": ["Request identifier."],
        "preconditions": ["Request is visible."],
        "ordered_steps": [
            {
                "step_id": "review_request",
                "action_intent": "Review the request.",
                "tool_name": "inspect_request",
                "argument_bindings": {"request_id": "REQ-1"},
                "preconditions": ["Request identifier is supplied."],
                "optional": False,
                "evidence_refs": ["visible request"],
            }
        ],
        "decision_hints": ["Escalate unusual scope."],
        "verification_steps": ["Confirm request status."],
        "stop_conditions": ["Stop when unavailable."],
        "escalation_hints": ["Escalate missing approval."],
        "source_trace_ids": ["trace_access_01"],
        "instruction_provenance": {"source": "demonstration"},
        "compiler_manifest_ref": "sha256:" + "a" * 64,
    }
    value.update(overrides)
    return value


def _script(
    steps: tuple[TransportResponse | str, ...],
    *,
    caps: ProviderCapabilities | None = None,
    pricing: TokenPricing | None = None,
    max_attempts: int = 3,
    reasoning_effort: str | None = None,
) -> ScriptedModelClient:
    return ScriptedModelClient(
        capabilities=caps or _caps(),
        script=steps,
        max_attempts=max_attempts,
        pricing=pricing,
        reasoning_effort=reasoning_effort,
    )  # type: ignore[arg-type]


def _run(
    client: ModelClient, request: ModelRequest | None = None, schema: type[BaseModel] = Output
) -> Any:
    return asyncio.run(client.structured(request or _request(), schema))


def _error(call: Any) -> ModelAdapterError:
    with pytest.raises(ModelAdapterError) as caught:
        call()
    return caught.value


def _hash(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _unchecked_model(target: type[BaseModel], **values: object) -> BaseModel:
    return BaseModel.model_construct.__func__(target, **values)


def _perfect_marker_forgery(target: type[BaseModel], **values: object) -> BaseModel:
    forged = _unchecked_model(target, **values)
    object.__setattr__(forged, "__pydantic_private__", {"_ssb_valid": True})
    return forged


def _assert_error(
    error: ModelAdapterError,
    code: str,
    attempts: int,
    before: bool,
    response_hash: str | None = None,
) -> None:
    assert (error.code, error.attempts, error.before_meaningful_behavior) == (
        code,
        attempts,
        before,
    )
    assert (error.raw_request_hash is not None) is (attempts > 0)
    assert error.raw_response_hash == response_hash


def test_public_symbols_field_order_and_generic_model_response() -> None:
    from shadowskillbench import models

    assert models.__all__ == [
        "AsyncTransport",
        "Message",
        "ModelAdapterError",
        "ModelClient",
        "ModelRequest",
        "ModelResponse",
        "OpenAICompatibleClient",
        "OllamaNativeClient",
        "OllamaNativeToolClient",
        "ProviderCapabilities",
        "ScriptedModelClient",
        "TokenCost",
        "TokenPricing",
        "TokenUsage",
        "TransportResponse",
        "TransportTerminalError",
        "TransportTransientError",
    ]
    assert tuple(Message.model_fields) == ("role", "content")
    assert tuple(ProviderCapabilities.model_fields) == (
        "provider",
        "model",
        "model_version",
        "supports_system_role",
        "supports_developer_role",
        "supports_seed",
        "supports_structured_output",
    )
    assert tuple(ModelRequest.model_fields) == ("messages", "temperature", "seed", "max_tokens")
    assert tuple(TokenUsage.model_fields) == ("input_tokens", "output_tokens", "total_tokens")
    assert tuple(TokenPricing.model_fields) == (
        "currency",
        "input_nanos_per_token",
        "output_nanos_per_token",
    )
    assert tuple(TokenCost.model_fields) == (
        "currency",
        "input_nanos_per_token",
        "output_nanos_per_token",
        "input_nanos",
        "output_nanos",
        "total_nanos",
    )
    assert tuple(TransportResponse.model_fields) == ("status_code", "body")
    assert tuple(ModelResponse.model_fields) == (
        "capabilities",
        "output",
        "raw_request_hash",
        "raw_response_hash",
        "structured_output_schema_hash",
        "usage",
        "cost",
        "attempts",
    )
    assert ModelResponse.__parameters__
    assert ModelResponse[Output].model_fields["output"].annotation is Output
    assert AsyncTransport is not None and ModelClient is not None


@pytest.mark.parametrize(
    ("factory", "values"),
    [
        (Message, {"role": "user", "content": b"x"}),
        (Message, {"role": "tool", "content": "x"}),
        (Message, {"role": "user", "content": "x\x00"}),
        (Message, {"role": "user", "content": "\ud800"}),
        (
            ProviderCapabilities,
            {
                "provider": " ",
                "model": "m",
                "model_version": "v",
                "supports_system_role": True,
                "supports_developer_role": True,
                "supports_seed": True,
                "supports_structured_output": True,
            },
        ),
        (
            ProviderCapabilities,
            {
                "provider": "p",
                "model": "m",
                "model_version": "v",
                "supports_system_role": 1,
                "supports_developer_role": True,
                "supports_seed": True,
                "supports_structured_output": True,
            },
        ),
        (ModelRequest, {"messages": (), "temperature": 0.0, "seed": None, "max_tokens": 1}),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": 0,
                "seed": None,
                "max_tokens": 1,
            },
        ),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": -0.0,
                "seed": None,
                "max_tokens": 1,
            },
        ),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": float("nan"),
                "seed": None,
                "max_tokens": 1,
            },
        ),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": 0.0,
                "seed": True,
                "max_tokens": 1,
            },
        ),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": 0.0,
                "seed": 2**63,
                "max_tokens": 1,
            },
        ),
        (
            ModelRequest,
            {
                "messages": [Message(role="user", content="x")],
                "temperature": 0.0,
                "seed": None,
                "max_tokens": False,
            },
        ),
        (TokenUsage, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4}),
        (TokenUsage, {"input_tokens": -1, "output_tokens": 0, "total_tokens": -1}),
        (
            TokenPricing,
            {"currency": "EUR", "input_nanos_per_token": 1, "output_nanos_per_token": 1},
        ),
        (
            TokenPricing,
            {"currency": "USD", "input_nanos_per_token": 1.0, "output_nanos_per_token": 1},
        ),
        (TransportResponse, {"status_code": True, "body": b""}),
        (TransportResponse, {"status_code": 200, "body": bytearray()}),
    ],
)
def test_public_values_enforce_exact_types_and_budgets(
    factory: Any, values: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(**values)


def test_message_and_request_budgets_and_blank_content() -> None:
    with pytest.raises(ValueError):
        Message(role="user", content="x" * 1_000_001)
    with pytest.raises(ValueError):
        ModelRequest(
            messages=tuple(Message(role="user", content="x" * 4_000) for _ in range(251)),
            temperature=0.0,
            seed=None,
            max_tokens=1,
        )
    with pytest.raises(ValueError):
        ModelRequest(
            messages=tuple(Message(role="user", content="x") for _ in range(257)),
            temperature=0.0,
            seed=None,
            max_tokens=1,
        )
    assert Message(role="assistant", content="   ").content == "   "


class ListSubclass(list[object]):
    pass


class DictSubclass(dict[str, object]):
    pass


class KeySubclass(str):
    pass


class HostileMapping(Mapping[str, object]):
    def __iter__(self):
        raise AssertionError("custom mappings must never be iterated")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        raise AssertionError("custom mappings must never be read")


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("list_subclass", ListSubclass([Message(role="user", content="x")])),
        (
            "dict_subclass",
            DictSubclass(
                {
                    "messages": (Message(role="user", content="x"),),
                    "temperature": 0.0,
                    "seed": None,
                    "max_tokens": 1,
                }
            ),
        ),
        ("hostile_mapping", HostileMapping()),
        (
            "subclass_key",
            {
                KeySubclass("messages"): (Message(role="user", content="x"),),
                "temperature": 0.0,
                "seed": None,
                "max_tokens": 1,
            },
        ),
    ],
)
def test_ingress_rejects_subclasses_custom_containers_and_subclass_keys(
    label: str, value: object
) -> None:
    del label
    with pytest.raises((TypeError, ValueError)):
        ModelRequest.model_validate(value)


def test_json_and_typeadapter_json_ingress_are_forbidden() -> None:
    message = Message(role="user", content="x")
    raw = {"messages": (message,), "temperature": 0.0, "seed": None, "max_tokens": 1}
    with pytest.raises((TypeError, ValueError)):
        ModelRequest.model_validate_json(json.dumps(raw, default=str))
    with pytest.raises((TypeError, ValueError)):
        TypeAdapter(ModelRequest).validate_json(
            '{"messages":[],"temperature":0,"seed":null,"max_tokens":1}'
        )


@pytest.mark.parametrize(
    "operation", ["construct", "model_copy", "copy", "deepcopy", "parse_raw", "parse_file"]
)
def test_unvalidated_copy_and_parse_paths_are_rejected(operation: str, tmp_path: Path) -> None:
    path = tmp_path / "message.json"
    path.write_text('{"role":"user","content":"x"}')
    valid = Message(role="user", content="x")
    call = {
        "construct": lambda: Message.model_construct(role="user", content="x"),
        "model_copy": lambda: valid.model_copy(update={"content": 1}),
        "copy": lambda: valid.copy(update={"content": 1}),
        "deepcopy": lambda: copy.deepcopy(_unchecked_model(Message, role="user", content="x")),
        "parse_raw": lambda: Message.parse_raw('{"role":"user","content":"x"}'),
        "parse_file": lambda: Message.parse_file(path),
    }[operation]
    with pytest.raises((TypeError, ValueError, AttributeError)):
        call()


def test_value_models_detach_caller_storage() -> None:
    messages = [Message(role="user", content="x")]
    request = ModelRequest(messages=messages, temperature=0.0, seed=None, max_tokens=1)
    messages.append(Message(role="assistant", content="changed"))
    assert len(request.messages) == 1


@pytest.mark.parametrize("kind", ["forged", "private", "extra", "field_set"])
def test_value_models_reject_forged_private_extra_and_field_set_state(kind: str) -> None:
    if kind == "forged":
        value = _unchecked_model(Message, role="user", content="x")
    elif kind == "private":
        value = Message(role="user", content="x")
        object.__setattr__(value, "__pydantic_private__", {"secret": "x"})
    elif kind == "extra":
        with pytest.raises((TypeError, ValueError)):
            Message.model_validate({"role": "user", "content": "x", "extra": "x"})
        return
    else:
        value = Message(role="user", content="x")
        object.__setattr__(value, "__pydantic_fields_set__", set())
    with pytest.raises((TypeError, ValueError)):
        ModelRequest(messages=(value,), temperature=0.0, seed=None, max_tokens=1)


class AbstractOutput(BaseModel, ABC):
    value: str

    @abstractmethod
    def required(self) -> str: ...


class GenericOutput[U](BaseModel):
    value: U


class NonObjectSchema(Output):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> object:
        return []


class NonCanonicalSchema(Output):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, object]:
        return {"bad": float("nan")}


class OversizeSchema(Output):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, object]:
        return {"payload": "x" * 1_000_001}


class ExplodingSchema(Output):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, object]:
        raise RuntimeError("SECRET_SCHEMA_CANARY")


class OverflowSchema(Output):
    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, object]:
        raise OverflowError("SECRET_SCHEMA_OVERFLOW")


@pytest.mark.parametrize(
    "schema",
    [
        BaseModel,
        object,
        AbstractOutput,
        GenericOutput,
        NonObjectSchema,
        NonCanonicalSchema,
        OversizeSchema,
        ExplodingSchema,
        OverflowSchema,
    ],
)
def test_schema_admission_is_exact_bounded_sanitized_and_pretransport(schema: object) -> None:
    client = _script((), max_attempts=1)
    error = _error(lambda: _run(client, schema=schema))
    _assert_error(error, "SCHEMA_ERROR", 0, True)
    assert client.recorded_request_bodies == ()
    assert "SECRET_SCHEMA_CANARY" not in str(error) + repr(error) + str(error.args)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None


def test_capability_then_request_then_schema_precedence_and_no_schema_call() -> None:
    client = _script((), caps=_caps(supports_seed=False), max_attempts=1)
    error = _error(lambda: _run(client, schema=ExplodingSchema))
    _assert_error(error, "CONFIGURATION_ERROR", 0, True)
    forged = _unchecked_model(ModelRequest, messages=(), temperature=0.0, seed=None, max_tokens=1)
    error = _error(lambda: _run(_script((), max_attempts=1), forged, Output))
    _assert_error(error, "CONFIGURATION_ERROR", 0, True)


def test_pinned_ssb_oai_chat1_request_schema_bytes_and_hashes() -> None:
    client = _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1)
    response = _run(client)
    expected = (
        b'{"max_tokens":33,"messages":[{"content":"return JSON","role":"user"}],'
        b'"model":"model-test",'
        b'"response_format":{"json_schema":{"name":"shadowskillbench_output","schema":{"properties":'
        b'{"value":{"title":"Value","type":"string"}},"required":["value"],"title":"Output","type":"object"},'
        b'"strict":true},"type":"json_schema"},"seed":7,"stream":false,"temperature":0.0}'
    )
    assert client.recorded_request_bodies == (expected,)
    assert response.raw_request_hash == _hash(expected)
    assert (
        response.structured_output_schema_hash
        == "sha256:7bd50aee661b52550a7990b12c8326b5ec7f1cd5f0bd5c77cfa757a80c840aca"
    )
    payload = json.loads(expected)
    assert set(payload) == {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "seed",
        "response_format",
    }
    assert payload["messages"] == [{"content": "return JSON", "role": "user"}]
    assert "reasoning_effort" not in payload
    seedless = _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1)
    _run(seedless, _request(seed=None))
    assert b'"seed"' not in seedless.recorded_request_bodies[0]
    assert all(x not in expected for x in (b"scripted", b"v1", b"USD", b"retry"))


def test_opt_in_reasoning_effort_binds_canonical_request_bytes_and_hash() -> None:
    client = _script(
        (TransportResponse(status_code=200, body=_body()),),
        max_attempts=1,
        reasoning_effort="none",
    )

    response = _run(client)
    payload = json.loads(client.recorded_request_bodies[0])

    assert client.reasoning_effort == "none"
    assert payload["reasoning_effort"] == "none"
    assert response.raw_request_hash == _hash(client.recorded_request_bodies[0])
    assert (
        client.recorded_request_bodies[0]
        != _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1)._preflight(
            _request(), Output
        )[2]
    )


@pytest.mark.parametrize("reasoning_effort", ("", "ultra", 1))
def test_reasoning_effort_rejects_invalid_configuration(reasoning_effort: object) -> None:
    with pytest.raises(ValueError, match="client configuration"):
        _script((), max_attempts=1, reasoning_effort=reasoning_effort)  # type: ignore[arg-type]


def test_raw_response_hash_preserves_whitespace_and_key_order() -> None:
    compact = _body()
    spaced = (
        b'{ "usage" : { "total_tokens" : 5 , "completion_tokens" : 3 , "prompt_tokens" : 2 },'
        b' "choices" : [{"message":{"content":"{\\"value\\":\\"ok\\"}","role":"assistant"},'
        b'"finish_reason":"stop","index":0}], "model":"model-test" }'
    )
    first = _run(_script((TransportResponse(status_code=200, body=compact),), max_attempts=1))
    second = _run(_script((TransportResponse(status_code=200, body=spaced),), max_attempts=1))
    assert (first.raw_response_hash, second.raw_response_hash) == (_hash(compact), _hash(spaced))
    assert first.raw_response_hash != second.raw_response_hash


def _choice(**message: object) -> list[dict[str, object]]:
    return [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}]


@pytest.mark.parametrize(
    "body",
    [
        b"\xff",
        b'{"model":"model-test","model":"model-test","choices":[],"usage":{}}',
        b'{"model":"model-test","choices":[],"usage":{"prompt_tokens":NaN}}',
        _body(model="wrong"),
        _body(choices=[]),
        _body(
            choices=[
                {
                    "index": False,
                    "message": {"role": "assistant", "content": '{"value":"ok"}'},
                    "finish_reason": "stop",
                }
            ]
        ),
        _body(
            choices=[
                {
                    "index": 0.0,
                    "message": {"role": "assistant", "content": '{"value":"ok"}'},
                    "finish_reason": "stop",
                }
            ]
        ),
        _body(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "user", "content": '{"value":"ok"}'},
                    "finish_reason": "stop",
                }
            ]
        ),
        _body(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"value":"ok"}'},
                    "finish_reason": "length",
                }
            ]
        ),
        _body(choices=_choice(content='{ "value":"ok" }', refusal="no")),
        _body(choices=_choice(content=1)),
        _body(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}),
        _body(usage={"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1}),
        _body(choices=_choice(content="[]")),
        _body(choices=_choice(content='{"value":"ok","value":"bad"}')),
        _body(choices=_choice(content='{"value":NaN}')),
    ],
)
def test_success_envelope_content_and_usage_are_strict_and_never_retry(body: bytes) -> None:
    client = _script(
        (
            TransportResponse(status_code=200, body=body),
            TransportResponse(status_code=200, body=_body()),
        ),
        max_attempts=2,
    )
    error = _error(lambda: _run(client))
    assert error.code == "MODEL_OUTPUT_INVALID" and error.attempts == 1
    assert error.raw_response_hash == _hash(body) and len(client.recorded_request_bodies) == 1


@pytest.mark.parametrize(
    ("body", "before"),
    [
        (_body(choices=_choice(content=None)), True),
        (_body(choices=_choice(content="   ")), True),
        (_body(model="wrong"), False),
        (
            _body(
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"value":"ok"}'},
                        "finish_reason": "length",
                    }
                ]
            ),
            False,
        ),
        (_body(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}), False),
        (_body(choices=_choice(content="{")), False),
        (_body(choices=_choice(content='{"value": 1}')), False),
    ],
)
def test_meaningful_behavior_matrix_marks_post_content_failures_false(
    body: bytes, before: bool
) -> None:
    error = _error(
        lambda: _run(_script((TransportResponse(status_code=200, body=body),), max_attempts=1))
    )
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, before, _hash(body))


def test_output_strict_extra_exact_class_and_schema_method_failures_are_sanitized() -> None:
    for content in ('{"value":7}', '{"value":"ok","extra":true}'):
        error = _error(
            lambda content=content: _run(
                _script(
                    (
                        TransportResponse(
                            status_code=200, body=_body(choices=_choice(content=content))
                        ),
                    ),
                    max_attempts=1,
                )
            )
        )
        assert (error.code, error.before_meaningful_behavior) == ("MODEL_OUTPUT_INVALID", False)

    class ExplodingOutput(Output):
        @classmethod
        def model_validate(cls, value: object, **kwargs: Any) -> Output:
            raise RuntimeError("SECRET_OUTPUT_CANARY")

    error = _error(
        lambda: _run(
            _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1),
            schema=ExplodingOutput,
        )
    )
    assert error.code == "MODEL_OUTPUT_INVALID"
    assert "SECRET_OUTPUT_CANARY" not in str(error) + repr(error) + str(error.args)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None

    class WrongClassOutput(Output):
        @classmethod
        def model_validate(cls, value: object, **kwargs: Any) -> OtherOutput:
            return OtherOutput(value="ok")

    error = _error(
        lambda: _run(
            _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1),
            schema=WrongClassOutput,
        )
    )
    assert (error.code, error.before_meaningful_behavior) == ("MODEL_OUTPUT_INVALID", False)


def test_pricing_no_price_exact_cost_and_overflow_semantics() -> None:
    assert (
        _run(_script((TransportResponse(status_code=200, body=_body()),), max_attempts=1)).cost
        is None
    )
    priced = _run(
        _script(
            (TransportResponse(status_code=200, body=_body()),),
            pricing=TokenPricing(currency="USD", input_nanos_per_token=2, output_nanos_per_token=3),
            max_attempts=1,
        )
    )
    assert priced.cost is not None and priced.cost.total_nanos == 13
    body = _body(
        usage={"prompt_tokens": 2**63 - 1, "completion_tokens": 0, "total_tokens": 2**63 - 1}
    )
    error = _error(
        lambda: _run(
            _script(
                (TransportResponse(status_code=200, body=body),),
                pricing=TokenPricing(
                    currency="USD", input_nanos_per_token=2, output_nanos_per_token=0
                ),
                max_attempts=1,
            )
        )
    )
    _assert_error(error, "CONFIGURATION_ERROR", 1, False, _hash(body))


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 599])
def test_transient_status_matrix_retries_with_exact_byte_identity(status: int) -> None:
    client = _script(
        (
            TransportResponse(status_code=status, body=b"transient"),
            TransportResponse(status_code=200, body=_body()),
        ),
        max_attempts=2,
    )
    response = _run(client)
    assert response.attempts == 2
    assert client.recorded_request_bodies[0] == client.recorded_request_bodies[1]
    assert response.raw_response_hash == _hash(_body())


@pytest.mark.parametrize(
    ("script", "expected", "expected_attempts"),
    [
        (("transient", "transient"), "MODEL_PROVIDER_TRANSIENT", 2),
        (("terminal",), "MODEL_PROVIDER_TERMINAL", 1),
    ],
)
def test_scripted_markers_and_exhaustion_have_exact_results(
    script: tuple[str, ...], expected: str, expected_attempts: int
) -> None:
    client = _script(script, max_attempts=2)
    error = _error(lambda: _run(client))
    _assert_error(error, expected, expected_attempts, True)
    assert len(client.recorded_request_bodies) == expected_attempts
    exhausted = _error(lambda: _run(_script((), max_attempts=1)))
    _assert_error(exhausted, "MODEL_PROVIDER_TERMINAL", 1, True)


def test_exception_retry_terminal_cancellation_and_final_hash_reset() -> None:
    calls: list[bytes] = []
    steps: list[object] = [
        TransportResponse(status_code=503, body=b"earlier"),
        TimeoutError("SECRET"),
        TransportResponse(status_code=200, body=_body()),
    ]

    async def transport(body: bytes) -> TransportResponse:
        calls.append(bytes(body))
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step  # type: ignore[return-value]

    response = _run(
        OpenAICompatibleClient(capabilities=_caps(), transport=transport, max_attempts=3)
    )
    assert response.attempts == 3 and calls[0] == calls[1] == calls[2]
    assert response.raw_response_hash == _hash(_body())

    async def terminal(_: bytes) -> TransportResponse:
        raise RuntimeError("SECRET_TRANSPORT_CANARY")

    error = _error(
        lambda: _run(
            OpenAICompatibleClient(capabilities=_caps(), transport=terminal, max_attempts=2)
        )
    )
    _assert_error(error, "MODEL_PROVIDER_TERMINAL", 1, True)
    assert "SECRET_TRANSPORT_CANARY" not in str(error) + repr(error) + str(error.args)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None

    async def cancelled(_: bytes) -> TransportResponse:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(OpenAICompatibleClient(capabilities=_caps(), transport=cancelled, max_attempts=1))

    async def interrupted(_: bytes) -> TransportResponse:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _run(OpenAICompatibleClient(capabilities=_caps(), transport=interrupted, max_attempts=1))


def test_adapter_error_public_args_attributes_and_slots_are_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ModelAdapterError(
        code="MODEL_PROVIDER_TERMINAL",
        attempts=1,
        before_meaningful_behavior=True,
        raw_request_hash="sha256:" + "a" * 64,
        raw_response_hash=None,
    )
    assert error.args == ("MODEL_PROVIDER_TERMINAL",)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None
    assert "SECRET_CANARY" not in repr(error) + str(error) + str(vars(error)) + caplog.text
    assert vars(error) == {}
    assert (
        error.code,
        error.attempts,
        error.before_meaningful_behavior,
        error.raw_request_hash,
        error.raw_response_hash,
    ) == ("MODEL_PROVIDER_TERMINAL", 1, True, "sha256:" + "a" * 64, None)


def test_scripted_client_rejects_hostile_markers_and_forged_responses_and_detaches_records() -> (
    None
):
    class Marker(str):
        pass

    with pytest.raises(ValueError):
        ScriptedModelClient(capabilities=_caps(), script=["transient"], max_attempts=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ScriptedModelClient(capabilities=_caps(), script=(Marker("transient"),), max_attempts=1)
    with pytest.raises(ValueError):
        ScriptedModelClient(capabilities=_caps(), script=(object(),), max_attempts=1)  # type: ignore[arg-type]
    forged = _unchecked_model(TransportResponse, status_code=200, body=_body())
    with pytest.raises(ValueError):
        ScriptedModelClient(capabilities=_caps(), script=(forged,), max_attempts=1)
    client = _script((), max_attempts=1)
    _error(lambda: _run(client))
    first = client.recorded_request_bodies
    assert type(first) is tuple and type(first[0]) is bytes
    assert first == client.recorded_request_bodies and first is not client.recorded_request_bodies


def test_perfect_marker_forged_capabilities_and_pricing_are_rejected() -> None:
    async def transport(_: bytes) -> TransportResponse:
        raise AssertionError("forged configuration must fail before transport")

    forged_capabilities = _perfect_marker_forgery(
        ProviderCapabilities,
        provider="scripted",
        model="model-test",
        model_version="v1",
        supports_system_role=True,
        supports_developer_role=True,
        supports_seed=True,
        supports_structured_output=True,
    )
    with pytest.raises(ValueError):
        OpenAICompatibleClient(
            capabilities=forged_capabilities,
            transport=transport,
            max_attempts=1,
        )

    forged_pricing = _perfect_marker_forgery(
        TokenPricing,
        currency="USD",
        input_nanos_per_token=2,
        output_nanos_per_token=3,
    )
    with pytest.raises(ValueError):
        OpenAICompatibleClient(
            capabilities=_caps(),
            transport=transport,
            max_attempts=1,
            pricing=forged_pricing,
        )


def test_perfect_marker_forged_message_and_request_are_rejected_before_transport() -> None:
    forged_message = _perfect_marker_forgery(Message, role="user", content="return JSON")
    with pytest.raises(ValueError):
        ModelRequest(
            messages=(forged_message,),
            temperature=0.0,
            seed=None,
            max_tokens=1,
        )

    forged_request = _perfect_marker_forgery(
        ModelRequest,
        messages=(Message(role="user", content="return JSON"),),
        temperature=0.0,
        seed=7,
        max_tokens=33,
    )
    client = _script((), max_attempts=1)
    error = _error(lambda: _run(client, forged_request))
    _assert_error(error, "CONFIGURATION_ERROR", 0, True)
    assert client.recorded_request_bodies == ()


def test_perfect_marker_forged_transport_response_is_terminal_without_response_hash() -> None:
    forged_response = _perfect_marker_forgery(TransportResponse, status_code=200, body=_body())
    with pytest.raises(ValueError):
        _script((forged_response,), max_attempts=1)

    async def transport(_: bytes) -> TransportResponse:
        return forged_response  # type: ignore[return-value]

    error = _error(
        lambda: _run(
            OpenAICompatibleClient(capabilities=_caps(), transport=transport, max_attempts=1)
        )
    )
    _assert_error(error, "MODEL_PROVIDER_TERMINAL", 1, True)


class DirectMessageSubclass(Message):
    pass


class DirectRequestSubclass(ModelRequest):
    pass


def test_type_adapter_python_ingress_is_exact_detached_and_rejects_forged_instances() -> None:
    raw_message = {"role": "user", "content": "original"}
    parsed_message = TypeAdapter(Message).validate_python(raw_message)
    raw_message["content"] = "changed"
    assert type(parsed_message) is Message and parsed_message.content == "original"

    raw_messages = [{"role": "user", "content": "original"}]
    parsed_request = TypeAdapter(ModelRequest).validate_python(
        {"messages": raw_messages, "temperature": 0.0, "seed": None, "max_tokens": 1}
    )
    raw_messages[0]["content"] = "changed"
    assert type(parsed_request) is ModelRequest
    assert parsed_request.messages[0].content == "original"

    owned_message = Message(role="user", content="exact")
    revalidated_message = TypeAdapter(Message).validate_python(owned_message)
    assert type(revalidated_message) is Message
    assert revalidated_message == owned_message and revalidated_message is not owned_message
    with pytest.raises((TypeError, ValueError)):
        TypeAdapter(Message).validate_python(
            _perfect_marker_forgery(Message, role="user", content="forged")
        )


def test_direct_message_and_request_subclasses_cannot_construct_validate_or_cross_boundary() -> (
    None
):
    with pytest.raises((TypeError, ValueError)):
        DirectMessageSubclass(role="user", content="x")
    with pytest.raises((TypeError, ValueError)):
        DirectMessageSubclass.model_validate({"role": "user", "content": "x"})
    with pytest.raises((TypeError, ValueError)):
        DirectRequestSubclass(
            messages=(Message(role="user", content="x"),),
            temperature=0.0,
            seed=None,
            max_tokens=1,
        )
    with pytest.raises((TypeError, ValueError)):
        DirectRequestSubclass.model_validate(
            {
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 0.0,
                "seed": None,
                "max_tokens": 1,
            }
        )

    forged_message_subclass = _perfect_marker_forgery(
        DirectMessageSubclass, role="user", content="x"
    )
    with pytest.raises((TypeError, ValueError)):
        ModelRequest(messages=(forged_message_subclass,), temperature=0.0, seed=None, max_tokens=1)
    forged_request_subclass = _perfect_marker_forgery(
        DirectRequestSubclass,
        messages=(Message(role="user", content="x"),),
        temperature=0.0,
        seed=None,
        max_tokens=1,
    )
    error = _error(lambda: _run(_script((), max_attempts=1), forged_request_subclass))
    _assert_error(error, "CONFIGURATION_ERROR", 0, True)


class OutputSubclass(Output):
    pass


def _model_response(output: BaseModel, cost: TokenCost | None = None) -> ModelResponse[Output]:
    return ModelResponse[Output](
        capabilities=_caps(),
        output=output,
        raw_request_hash="sha256:" + "a" * 64,
        raw_response_hash="sha256:" + "b" * 64,
        structured_output_schema_hash="sha256:" + "c" * 64,
        usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        cost=cost,
        attempts=1,
    )


def test_model_response_generic_rejects_output_subclass_and_forged_output_and_binds_cost() -> None:
    cost = TokenCost(
        currency="USD",
        input_nanos_per_token=2,
        output_nanos_per_token=3,
        input_nanos=4,
        output_nanos=9,
        total_nanos=13,
    )
    accepted = _model_response(Output(value="ok"), cost)
    assert type(accepted.output) is Output and accepted.cost == cost
    with pytest.raises((TypeError, ValueError)):
        _model_response(OutputSubclass(value="ok"), cost)
    forged_output = _unchecked_model(Output, value=1)
    with pytest.raises((TypeError, ValueError)):
        _model_response(forged_output, cost)
    mismatched_cost = TokenCost(
        currency="USD",
        input_nanos_per_token=2,
        output_nanos_per_token=3,
        input_nanos=1,
        output_nanos=2,
        total_nanos=3,
    )
    with pytest.raises((TypeError, ValueError)):
        _model_response(Output(value="ok"), mismatched_cost)
    from_client = _run(_script((TransportResponse(status_code=200, body=_body()),), max_attempts=1))
    assert type(from_client) is ModelResponse[Output]
    assert type(from_client.output) is Output


class CodeSubclass(str):
    pass


class AdapterErrorSubclass(ModelAdapterError):
    pass


def _adapter_error_values(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "code": "MODEL_PROVIDER_TERMINAL",
        "attempts": 1,
        "before_meaningful_behavior": True,
        "raw_request_hash": "sha256:" + "a" * 64,
        "raw_response_hash": None,
    }
    values.update(changes)
    return values


@pytest.mark.parametrize(
    "changes",
    [
        {"code": CodeSubclass("MODEL_PROVIDER_TERMINAL")},
        {"code": "NOT_A_D025_CODE"},
        {"attempts": 6},
        {"attempts": 1, "raw_request_hash": None},
        {"attempts": 1, "raw_request_hash": None, "raw_response_hash": "sha256:" + "b" * 64},
    ],
)
def test_model_adapter_error_rejects_nonexact_codes_and_inconsistent_hashes(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelAdapterError(**_adapter_error_values(**changes))  # type: ignore[arg-type]


def test_model_adapter_error_is_exact_immutable_and_code_only() -> None:
    with pytest.raises((TypeError, ValueError)):
        AdapterErrorSubclass(**_adapter_error_values())
    error = ModelAdapterError(**_adapter_error_values())
    for field, value in (
        ("code", "SECRET_CODE"),
        ("attempts", 0),
        ("args", ("SECRET_ARGS",)),
        ("secret", "SECRET_ATTRIBUTE"),
    ):
        with pytest.raises((AttributeError, TypeError)):
            setattr(error, field, value)
    assert error.args == ("MODEL_PROVIDER_TERMINAL",)
    assert str(error) == "MODEL_PROVIDER_TERMINAL"
    assert repr(error) == "ModelAdapterError(code='MODEL_PROVIDER_TERMINAL')"
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None
    assert "SECRET" not in str(error) + repr(error) + str(vars(error))


@pytest.mark.parametrize(
    ("content", "refusal", "before"),
    [
        (None, "refusal", False),
        (None, "   ", True),
        (None, 1, True),
        ('{"value":"ok"}', 1, False),
    ],
)
def test_meaningful_refusal_matrix(content: object, refusal: object, before: bool) -> None:
    body = _body(choices=_choice(content=content, refusal=refusal))
    error = _error(
        lambda: _run(_script((TransportResponse(status_code=200, body=body),), max_attempts=1))
    )
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, before, _hash(body))


def test_multiple_choice_envelope_with_extractable_nonblank_behavior_is_after_meaningful() -> None:
    body = _body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": None, "refusal": "declined"},
                "finish_reason": "stop",
            },
        ]
    )
    error = _error(
        lambda: _run(_script((TransportResponse(status_code=200, body=body),), max_attempts=1))
    )
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, False, _hash(body))


@pytest.mark.parametrize("final", [TimeoutError("SECRET_TIMEOUT"), RuntimeError("SECRET_TERMINAL")])
def test_final_transport_exception_resets_earlier_transient_response_hash(
    final: BaseException,
) -> None:
    steps: list[object] = [TransportResponse(status_code=503, body=b"earlier"), final]

    async def transport(_: bytes) -> TransportResponse:
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step  # type: ignore[return-value]

    error = _error(
        lambda: _run(
            OpenAICompatibleClient(capabilities=_caps(), transport=transport, max_attempts=2)
        )
    )
    expected = (
        "MODEL_PROVIDER_TRANSIENT" if isinstance(final, TimeoutError) else "MODEL_PROVIDER_TERMINAL"
    )
    _assert_error(error, expected, 2, True)


@pytest.mark.parametrize(
    "returned",
    [object(), _perfect_marker_forgery(TransportResponse, status_code=200, body=_body())],
)
def test_wrong_or_perfect_marker_forged_return_has_no_response_hash(returned: object) -> None:
    async def transport(_: bytes) -> TransportResponse:
        return returned  # type: ignore[return-value]

    error = _error(
        lambda: _run(
            OpenAICompatibleClient(capabilities=_caps(), transport=transport, max_attempts=1)
        )
    )
    _assert_error(error, "MODEL_PROVIDER_TERMINAL", 1, True)


def test_client_constructor_signatures_and_public_numeric_body_bounds() -> None:
    for client_type, names in (
        (
            OpenAICompatibleClient,
            ("capabilities", "transport", "max_attempts", "pricing", "reasoning_effort"),
        ),
        (
            ScriptedModelClient,
            ("capabilities", "script", "max_attempts", "pricing", "reasoning_effort"),
        ),
    ):
        parameters = signature(client_type).parameters
        assert tuple(parameters) == names
        assert all(parameter.kind is Parameter.KEYWORD_ONLY for parameter in parameters.values())
        assert parameters["pricing"].default is None
        assert parameters["reasoning_effort"].default is None
    with pytest.raises(TypeError):
        OpenAICompatibleClient(_caps(), lambda _: None, 1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        TransportResponse(status_code=200, body=b"x" * 4_000_001)
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=2**63, output_tokens=0, total_tokens=2**63)
    with pytest.raises(ValueError):
        TokenPricing(currency="USD", input_nanos_per_token=2**63, output_nanos_per_token=0)
    with pytest.raises(ValueError):
        TokenCost(
            currency="USD",
            input_nanos_per_token=0,
            output_nanos_per_token=0,
            input_nanos=2**63,
            output_nanos=0,
            total_nanos=2**63,
        )


class PrivateOutput(BaseModel):
    value: str
    _receipt: str = PrivateAttr(default="private")


class DefaultedOutput(BaseModel):
    value: str = "default"


class AliasedOutput(BaseModel):
    value: str = Field(alias="wire_value")


class FinalResponseFailureOutput(BaseModel):
    value: str
    _validation_calls: ClassVar[int] = 0

    @model_validator(mode="after")
    def fail_only_during_final_response_revalidation(self) -> FinalResponseFailureOutput:
        type(self)._validation_calls += 1
        if type(self)._validation_calls == 3:
            raise ValueError("SECRET_FINAL_RESPONSE_CANARY")
        return self


class SameInstanceFallbackOutput(BaseModel):
    model_config = ConfigDict(revalidate_instances="never")

    value: tuple[str, ...]

    @field_validator("value", mode="before")
    @classmethod
    def _wire_lists_only(cls, value: object) -> tuple[str, ...]:
        if type(value) is not list:
            raise ValueError("value must be a wire list")
        return tuple(value)


def test_model_response_supports_exact_skill_ir_after_strict_mapping_fallback() -> None:
    raw = _skill_ir_raw()
    output = SkillIR.model_validate(raw)
    direct = ModelResponse[SkillIR](
        capabilities=_caps(),
        output=output,
        raw_request_hash="sha256:" + "a" * 64,
        raw_response_hash="sha256:" + "b" * 64,
        structured_output_schema_hash="sha256:" + "c" * 64,
        usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        cost=None,
        attempts=1,
    )
    assert type(direct.output) is SkillIR
    assert direct.output is not output
    assert type(direct.output.ordered_steps) is tuple

    body = _body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(raw)},
                "finish_reason": "stop",
            }
        ]
    )
    response = _run(
        _script((TransportResponse(status_code=200, body=body),), max_attempts=1), schema=SkillIR
    )
    assert type(response) is ModelResponse[SkillIR]
    assert type(response.output) is SkillIR
    assert response.output.skill_id == "skill_adapter_compatibility"


def test_model_response_rejects_invalid_tampered_or_malformed_skill_ir_and_subclasses() -> None:
    invalid_body = _body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(_skill_ir_raw(domain="invalid")),
                },
                "finish_reason": "stop",
            }
        ]
    )
    error = _error(
        lambda: _run(
            _script((TransportResponse(status_code=200, body=invalid_body),), max_attempts=1),
            schema=SkillIR,
        )
    )
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, False, _hash(invalid_body))

    tampered = SkillIR.model_validate(_skill_ir_raw())
    object.__getattribute__(tampered, "__dict__")["domain"] = "invalid"
    with pytest.raises(ValueError):
        ModelResponse[SkillIR](
            capabilities=_caps(),
            output=tampered,
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            structured_output_schema_hash="sha256:" + "c" * 64,
            usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            cost=None,
            attempts=1,
        )

    malformed = SkillIR.model_validate(_skill_ir_raw())
    del object.__getattribute__(malformed, "__dict__")["objective"]
    with pytest.raises(ValueError):
        ModelResponse[SkillIR](
            capabilities=_caps(),
            output=malformed,
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            structured_output_schema_hash="sha256:" + "c" * 64,
            usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            cost=None,
            attempts=1,
        )

    class SkillIRSubclass(SkillIR):
        pass

    with pytest.raises(ValueError):
        ModelResponse[SkillIR](
            capabilities=_caps(),
            output=object.__new__(SkillIRSubclass),
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            structured_output_schema_hash="sha256:" + "c" * 64,
            usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            cost=None,
            attempts=1,
        )


def test_model_response_rejects_same_instance_fallback_output() -> None:
    with pytest.raises(ValueError):
        ModelResponse[SameInstanceFallbackOutput](
            capabilities=_caps(),
            output=SameInstanceFallbackOutput.model_validate({"value": ["wire"]}),
            raw_request_hash="sha256:" + "a" * 64,
            raw_response_hash="sha256:" + "b" * 64,
            structured_output_schema_hash="sha256:" + "c" * 64,
            usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            cost=None,
            attempts=1,
        )


def test_private_and_defaulted_caller_output_schemas_succeed_end_to_end() -> None:
    private_response = _run(
        _script((TransportResponse(status_code=200, body=_body()),), max_attempts=1),
        schema=PrivateOutput,
    )
    assert type(private_response) is ModelResponse[PrivateOutput]
    assert type(private_response.output) is PrivateOutput
    assert private_response.output.value == "ok"
    assert private_response.output._receipt == "private"

    assert DefaultedOutput().model_fields_set == set()
    defaulted_response = _run(
        _script(
            (TransportResponse(status_code=200, body=_body(choices=_choice(content="{}"))),),
            max_attempts=1,
        ),
        schema=DefaultedOutput,
    )
    assert type(defaulted_response) is ModelResponse[DefaultedOutput]
    assert type(defaulted_response.output) is DefaultedOutput
    assert defaulted_response.output.value == "default"

    aliased_response = _run(
        _script(
            (
                TransportResponse(
                    status_code=200,
                    body=_body(choices=_choice(content='{"wire_value":"aliased"}')),
                ),
            ),
            max_attempts=1,
        ),
        schema=AliasedOutput,
    )
    assert type(aliased_response) is ModelResponse[AliasedOutput]
    assert type(aliased_response.output) is AliasedOutput
    assert aliased_response.output.value == "aliased"


def test_final_model_response_construction_failure_is_sanitized_and_not_retried() -> None:
    FinalResponseFailureOutput._validation_calls = 0
    body = _body()
    client = _script(
        (
            TransportResponse(status_code=200, body=body),
            TransportResponse(status_code=200, body=_body()),
        ),
        max_attempts=2,
    )
    error = _error(lambda: _run(client, schema=FinalResponseFailureOutput))
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, False, _hash(body))
    assert len(client.recorded_request_bodies) == 1
    assert "SECRET_FINAL_RESPONSE_CANARY" not in str(error) + repr(error) + str(error.args)
    assert BaseException.__getattribute__(error, "__cause__") is None
    assert BaseException.__getattribute__(error, "__context__") is None


def test_invalid_output_carries_finish_reason_and_reported_usage_from_a_200_body() -> None:
    truncated = _body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }
        ],
        usage={"prompt_tokens": 65735, "completion_tokens": 16384, "total_tokens": 82119},
    )
    error = _error(
        lambda: _run(_script((TransportResponse(status_code=200, body=truncated),), max_attempts=1))
    )
    _assert_error(error, "MODEL_OUTPUT_INVALID", 1, True, _hash(truncated))
    assert error.finish_reason == "length"
    assert error.reported_usage == TokenUsage(
        input_tokens=65735, output_tokens=16384, total_tokens=82119
    )
    assert str(error) == "MODEL_OUTPUT_INVALID" and "length" not in repr(error)

    expected_usage = TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5)
    blank_finish = _choice(content="{")
    blank_finish[0]["finish_reason"] = "  "
    for body, expected in (
        (b"not-json", (None, None)),
        (_body(choices=_choice(content="{")), ("stop", expected_usage)),
        (
            _body(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}),
            ("stop", None),
        ),
        (_body(choices=blank_finish), (None, expected_usage)),
    ):
        error = _error(
            lambda body=body: _run(
                _script((TransportResponse(status_code=200, body=body),), max_attempts=1)
            )
        )
        assert error.code == "MODEL_OUTPUT_INVALID"
        assert (error.finish_reason, error.reported_usage) == expected

    terminal = _error(lambda: _run(_script(("terminal",), max_attempts=1)))
    assert (terminal.finish_reason, terminal.reported_usage) == (None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_response_hash": "sha256:" + "b" * 64, "finish_reason": "   "},
        {"raw_response_hash": "sha256:" + "b" * 64, "finish_reason": "x" * 33},
        {"raw_response_hash": "sha256:" + "b" * 64, "finish_reason": CodeSubclass("stop")},
        {"raw_response_hash": "sha256:" + "b" * 64, "reported_usage": {"input_tokens": 1}},
        {"raw_response_hash": None, "finish_reason": "length"},
        {
            "raw_response_hash": None,
            "reported_usage": TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        },
    ],
)
def test_model_adapter_error_rejects_invalid_response_diagnostics(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelAdapterError(**_adapter_error_values(**changes))  # type: ignore[arg-type]


def test_model_adapter_error_response_diagnostics_default_absent_and_are_immutable() -> None:
    error = ModelAdapterError(**_adapter_error_values())
    assert (error.finish_reason, error.reported_usage) == (None, None)
    with pytest.raises((AttributeError, TypeError)):
        error.finish_reason = "length"
    with pytest.raises((AttributeError, TypeError)):
        error.reported_usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
