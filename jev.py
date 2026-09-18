"""`@jev`: turn a Python function into a Jev (System One) structured-decision call.

The decorated function is never executed; it is the *specification* of a query:

- The docstring is a Jinja2 template. At call time it is rendered with the
  bound arguments and sent as the System One ``state``.
- The return annotation must be a ``pydantic.BaseModel`` subclass. Each field
  becomes a typed question:

  ================================  ======  ==============================
  Field type                        Jev     Coerced back as
  ================================  ======  ==============================
  ``bool``                          Noul    ``probability_yes >= 0.5``
  ``Literal[...]`` / ``Enum``       Choice  the selected label / member
  ``int`` with ``Field(ge, le)``    Score   ``lo + round(expected_score)``
  ``float`` with ``Field(ge, le)``  Score   linear interpolation over levels
  ================================  ======  ==============================

  Field ``description`` becomes the question's instructions. Score level
  labels default to the numbers in range; override them with
  ``Field(..., json_schema_extra={"levels": [...]})``.
- The answers are validated and returned as an instance of the return model.

The body always runs, and there are three useful things it can do:

- ``return fn.state()``: the body-less form; the rendered docstring is the
  whole state. A bare ``...`` or ``raise NotImplementedError`` also works.
- ``return fn.state(value)``: the body builds the state with full Python
  (loops, conditionals, f-strings, even ``await``). The decorated function's
  ``state`` method is typed ``value -> return-annotation``, so the body's
  return type-checks; the wrapper unwraps the marker and sends ``value`` as
  the state exactly as returned (the docstring is documentation in this
  form and is not sent). ``builder(fn)`` is the pure state-builder and can
  be unit-tested without any API call.
- ``return Model(...)``: construct the answer yourself and the API call is
  skipped entirely (a mock seam for tests).

Works on sync and async functions, bare (``@jev``) or configured
(``@jev(model=..., client=...)``). The client defaults to the
``TYPESAFE_API_KEY`` environment variable and ``jev-latest``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import typing
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, ParamSpec, Protocol, Self, TypeIs, TypeVar, cast, get_args, get_origin, overload

import annotated_types
import jinja2
from pydantic import BaseModel

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo
    from typesafe_sdk import ChoiceAnswer, JSONContent, NoulAnswer, ScoreAnswer, SystemOneResponse

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score, TypeSafeClient

__all__ = ["jev", "JevFn", "AsyncJevFn", "JevModel", "state_payload", "builder"]

P = ParamSpec("P")
R = TypeVar("R", bound=BaseModel)
class JevFn(Protocol[P, R]):
    """A sync ``@jev``-decorated function: call it to query Jev.

    ``state`` builds the state marker a body returns via ``return fn.state(...)``.
    ``map(items)`` applies the function to each item in a single batched call.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def state(self, value: JSONContent | None = None) -> R: ...
    def map(self, items: Sequence[JSONContent]) -> list[R]: ...


class AsyncJevFn(Protocol[P, R]):
    """An async ``@jev``-decorated function: await it to query Jev."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]: ...
    def state(self, value: JSONContent | None = None) -> R: ...
    def map(self, items: Sequence[JSONContent]) -> Coroutine[Any, Any, list[R]]: ...


class _JevDecorator(Protocol):
    """The configured ``@jev(...)`` form: preserves params, swaps the return."""

    @overload
    def __call__(self, fn: Callable[P, R], /) -> JevFn[P, R]: ...
    @overload
    def __call__(self, fn: Callable[P, Awaitable[R]], /) -> AsyncJevFn[P, R]: ...

# Attribute on the marker instance that carries the body-built state value.
_STATE_ATTR = "__jev_state__"
# Distinguishes "not a marker" from "marker carrying None" (the body-less form).
_ABSENT: Any = object()


class _StateBuilder(Generic[R]):
    """The ``state`` method on a ``@jev`` wrapper; builds the marker it unwraps."""

    def __init__(self, model: type[R]) -> None:
        self._model = model

    def __call__(self, value: JSONContent | None = None) -> R:
        # model_construct() is typed `-> Self`, so the marker is a genuine `R`
        # as far as any type checker is concerned; the payload rides along in a
        # dunder attribute and is unwrapped by the @jev wrapper before the
        # instance can be observed. A missing payload (None is never a valid
        # state) means "body-less": the rendered docstring is the state.
        marker = self._model.model_construct()
        object.__setattr__(marker, _STATE_ATTR, value)
        return marker


def state_payload(marker: BaseModel) -> JSONContent | None:
    """The value carried by a ``fn.state(...)`` marker.

    Useful for unit-testing state builders without making an API call.
    """
    value = getattr(marker, _STATE_ATTR, _ABSENT)
    if value is _ABSENT:
        raise TypeError(
            f"state_payload: {type(marker).__name__} is not a fn.state(...) marker"
        )
    return value


_builder_registry: weakref.WeakKeyDictionary[Callable[..., Any], Callable[..., Any]] = (
    weakref.WeakKeyDictionary()
)


def builder(fn: Callable[P, Any]) -> Callable[P, Any]:
    """The original function behind a ``@jev`` wrapper: the pure state-builder.

    ``builder(fn)(*args, **kwargs)`` runs the body without touching the API,
    so state construction can be unit-tested directly.
    """
    return _builder_registry.get(fn, fn)

# Jev's maximum choice cardinality.
_MAX_CHOICE_OPTIONS = 255
_MAX_SCORE_LEVELS = 256

# Noul -> bool coercion: p(yes) >= threshold. Overridable per function with
# @jev(bool_threshold=...) or globally with the env var.
_DEFAULT_BOOL_THRESHOLD = 0.5
_BOOL_THRESHOLD_ENV = "JEV_BOOL_THRESHOLD"

_jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)


# ---------------------------------------------------------------------------
# Public, fully-overloaded decorator
# ---------------------------------------------------------------------------


@overload
def jev(fn: Callable[P, R], /) -> JevFn[P, R]:
    """Bare `@jev` on a sync function."""
    ...


@overload
def jev(fn: Callable[P, Awaitable[R]], /) -> AsyncJevFn[P, R]:
    """Bare `@jev` on an async function."""
    ...


@overload
def jev(
    fn: None = None,
    /,
    *,
    model: str | None = None,
    client: TypeSafeClient | AsyncTypeSafeClient | None = None,
    bool_threshold: float | None = None,
) -> _JevDecorator:
    """Configured `@jev(...)`; preserves params, adds ``.state``."""
    ...


def jev(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    model: str | None = None,
    client: TypeSafeClient | AsyncTypeSafeClient | None = None,
    bool_threshold: float | None = None,
) -> Any:
    """Decorate `fn` so calling it queries Jev instead of running its body."""

    def decorator(func: Callable[..., Any]) -> Any:
        return _decorate(func, model=model, client=client, bool_threshold=bool_threshold)

    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Decoration-time compilation (fail fast, before any API call)
# ---------------------------------------------------------------------------

@dataclass
class _AnswersView:
    """The slice of a response the extractors read: a whole response, or a
    per-item view over a batched (map) response."""

    nouls: dict[str, NoulAnswer]
    choices: dict[str, ChoiceAnswer]
    scores: dict[str, ScoreAnswer]

    @classmethod
    def whole(cls, response: SystemOneResponse) -> Self:
        return cls(response.nouls, response.choices, response.scores)

    @classmethod
    def for_item(cls, response: SystemOneResponse, index: int) -> Self:
        prefix = f"{index}:"
        return cls(
            {k[len(prefix):]: a for k, a in response.nouls.items() if k.startswith(prefix)},
            {k[len(prefix):]: a for k, a in response.choices.items() if k.startswith(prefix)},
            {k[len(prefix):]: a for k, a in response.scores.items() if k.startswith(prefix)},
        )


# A per-field extractor: answers -> the value for that model field.
_Extractor = Callable[[_AnswersView], Any]


def _decorate(
    func: Callable[..., Any],
    *,
    model: str | None,
    client: TypeSafeClient | AsyncTypeSafeClient | None,
    bool_threshold: float | None,
) -> Any:
    return_model = _return_model_of(func)
    signature = inspect.signature(func)
    template = _compile_template(func)
    questions, extractors = _compile_questions(return_model, bool_threshold)

    if inspect.iscoroutinefunction(func):
        if client is not None and not isinstance(client, AsyncTypeSafeClient):
            raise TypeError(
                f"@jev: {func.__qualname__} is async and needs an AsyncTypeSafeClient, "
                f"got {type(client).__name__}"
            )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            body_result = await _acall_body(func, args, kwargs)
            override, state = _resolve_body(
                func, return_model, template, signature, args, kwargs, body_result
            )
            if override is not None:
                return override
            async_client = client if client is not None else _default_async_client()
            response = await async_client.system_one(state=state, questions=questions, model=model)
            return _materialize(return_model, extractors, response)

        setattr(async_wrapper, "state", _StateBuilder(return_model))

        async def amap_impl(items: Sequence[JSONContent]) -> Any:
            overrides, states, batched, state_array = await _map_prepare_async(
                func, return_model, template, signature, questions, items
            )
            if states:
                async_client = client if client is not None else _default_async_client()
                response = await async_client.system_one(
                    state=state_array, questions=batched, model=model
                )
            else:
                response = None
            return _map_finish(return_model, extractors, response, overrides, states, len(items))

        setattr(async_wrapper, "map", amap_impl)
        _builder_registry[async_wrapper] = func
        return async_wrapper

    if client is not None and not isinstance(client, TypeSafeClient):
        raise TypeError(
            f"@jev: {func.__qualname__} is sync and needs a TypeSafeClient, "
            f"got {type(client).__name__}"
        )

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        body_result = _call_body(func, args, kwargs)
        override, state = _resolve_body(
            func, return_model, template, signature, args, kwargs, body_result
        )
        if override is not None:
            return override
        sync_client = client if client is not None else _default_sync_client()
        response = sync_client.system_one(state=state, questions=questions, model=model)
        return _materialize(return_model, extractors, response)

    setattr(sync_wrapper, "state", _StateBuilder(return_model))

    def map_impl(items: Sequence[JSONContent]) -> Any:
        overrides, states, batched, state_array = _map_prepare(
            func, return_model, template, signature, questions, items
        )
        if states:
            sync_client = client if client is not None else _default_sync_client()
            response = sync_client.system_one(state=state_array, questions=batched, model=model)
        else:
            response = None
        return _map_finish(return_model, extractors, response, overrides, states, len(items))

    setattr(sync_wrapper, "map", map_impl)
    _builder_registry[sync_wrapper] = func
    return sync_wrapper


def _return_model_of(func: Callable[..., Any]) -> type[BaseModel]:
    annotation = typing.get_type_hints(func).get("return")
    if annotation is None:
        raise TypeError(f"@jev: {func.__qualname__} must declare a return annotation")
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        raise TypeError(
            f"@jev: {func.__qualname__} must return a pydantic BaseModel subclass, "
            f"got {annotation!r}"
        )
    return annotation


def _compile_template(func: Callable[..., Any]) -> jinja2.Template | None:
    doc = inspect.getdoc(func)
    if doc is None:
        return None
    try:
        return _jinja_env.from_string(doc)
    except jinja2.TemplateSyntaxError as exc:
        raise TypeError(
            f"@jev: the docstring of {func.__qualname__} is not a valid Jinja2 template: {exc}"
        ) from exc


def _compile_questions(
    return_model: type[BaseModel],
    bool_threshold: float | None,
) -> tuple[dict[str, Noul | Choice | Score], dict[str, _Extractor]]:
    questions: dict[str, Noul | Choice | Score] = {}
    extractors: dict[str, _Extractor] = {}
    for name, field in return_model.model_fields.items():
        question, extractor = _compile_field(return_model, name, field, bool_threshold)
        questions[name] = question
        extractors[name] = extractor
    return questions, extractors


def _compile_field(
    return_model: type[BaseModel],
    name: str,
    field: FieldInfo,
    bool_threshold: float | None,
) -> tuple[Noul | Choice | Score, _Extractor]:
    annotation = field.annotation
    instructions = field.description or name.replace("_", " ")

    # Order matters: `bool` is a subclass of `int`, and `Literal` checks must
    # precede the generic type checks.
    if annotation is bool:
        return _compile_noul(name, instructions, bool_threshold)

    literal_values = _literal_values(annotation)
    if literal_values is not None:
        return _compile_choice(name, instructions, _label_map(name, [(str(v), v) for v in literal_values]))

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _compile_choice(
            name, instructions, _label_map(name, [(str(m.value), m) for m in annotation])
        )

    if annotation is int:
        return _compile_score(return_model, name, field, is_integer=True, instructions=instructions)
    if annotation is float:
        return _compile_score(return_model, name, field, is_integer=False, instructions=instructions)

    raise TypeError(
        f"@jev: {return_model.__name__}.{name} has unsupported type {annotation!r}. "
        "Jev cannot generate strings, so fields must be bool, Literal[...], Enum, "
        "or int/float constrained with Field(ge=..., le=...)."
    )


def _validate_bool_threshold(value: float, source: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"@jev: bool threshold from {source} must be in [0, 1], got {value}")
    return value


def _bool_threshold(explicit: float | None) -> float:
    """Resolve the effective threshold: decorator arg > env var > default."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(_BOOL_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_BOOL_THRESHOLD
    try:
        return _validate_bool_threshold(float(raw), _BOOL_THRESHOLD_ENV)
    except ValueError:
        raise ValueError(
            f"@jev: {_BOOL_THRESHOLD_ENV} must be a float in [0, 1], got {raw!r}"
        ) from None


def _compile_noul(
    name: str, instructions: str, bool_threshold: float | None
) -> tuple[Noul, _Extractor]:
    # An explicit threshold is validated at decoration time; the env var is
    # resolved per call so tests and workers can tune it without re-importing.
    if bool_threshold is not None:
        _validate_bool_threshold(bool_threshold, "@jev(bool_threshold=...)")

    def extract(r: _AnswersView) -> bool:
        return r.nouls[name].noul >= _bool_threshold(bool_threshold)

    return Noul(instructions=instructions), extract


def _literal_values(annotation: Any) -> tuple[Any, ...] | None:
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    return None


def _is_list(value: Any) -> TypeIs[list[Any]]:
    return isinstance(value, list)


def _is_str_dict(value: Any) -> TypeIs[dict[str, Any]]:
    # pydantic's json_schema_extra is either a JsonDict or a callable, which
    # this excludes; keys are checked so the TypeIs is honest. (The cast turns
    # the isinstance narrowing's dict[Unknown, Unknown] into Any-typed keys.)
    return isinstance(value, dict) and all(
        isinstance(k, str) for k in cast("dict[Any, Any]", value)
    )


def _label_map(name: str, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Stringified label -> field value, rejecting collisions (e.g. Literal[1, "1"]).

    The check must happen here, on the pairs: a dict comprehension would
    silently dedupe colliding labels before anyone could count them.
    """
    label_to_value: dict[str, Any] = {}
    for label, value in pairs:
        if label in label_to_value:
            raise TypeError(
                f"@jev: field {name!r} has options that collide when stringified: {label!r}"
            )
        label_to_value[label] = value
    return label_to_value


def _compile_choice(
    name: str, instructions: str, label_to_value: dict[str, Any]
) -> tuple[Choice, _Extractor]:
    labels = list(label_to_value)
    if len(labels) > _MAX_CHOICE_OPTIONS:
        raise TypeError(
            f"@jev: field {name!r} has {len(labels)} options; "
            f"Jev supports at most {_MAX_CHOICE_OPTIONS} per choice"
        )
    question = Choice(instructions=instructions, criteria=dict.fromkeys(labels))
    return question, lambda r: label_to_value[r.choices[name].choice]


def _compile_score(
    return_model: type[BaseModel],
    name: str,
    field: FieldInfo,
    *,
    is_integer: bool,
    instructions: str,
) -> tuple[Score, _Extractor]:
    lo: Any = None
    hi: Any = None
    for constraint in field.metadata:
        if isinstance(constraint, annotated_types.Ge):
            lo = constraint.ge
        elif isinstance(constraint, annotated_types.Le):
            hi = constraint.le
    if lo is None or hi is None:
        raise TypeError(
            f"@jev: {return_model.__name__}.{name} is a score; "
            "constrain it with Field(ge=..., le=...)"
        )

    extra: dict[str, Any] = field.json_schema_extra if _is_str_dict(field.json_schema_extra) else {}
    raw_levels: Any = extra.get("levels")
    if raw_levels is not None and not _is_list(raw_levels):
        raise TypeError(
            f"@jev: {return_model.__name__}.{name}: "
            "json_schema_extra['levels'] must be a list of labels"
        )
    custom_levels: list[Any] | None = raw_levels

    if is_integer:
        lo_i, hi_i = int(lo), int(hi)
        if custom_levels is not None:
            levels = [str(x) for x in custom_levels]
        else:
            levels = [str(v) for v in range(lo_i, hi_i + 1)]
        if len(levels) != hi_i - lo_i + 1:
            raise TypeError(
                f"@jev: {return_model.__name__}.{name}: custom levels must have exactly "
                f"ge..le entries ({hi_i - lo_i + 1}), got {len(levels)}"
            )

        def extract_int(r: _AnswersView) -> int:
            return lo_i + int(round(r.scores[name].score))

        extractor: _Extractor = extract_int
    else:
        lo_f, hi_f = float(lo), float(hi)
        if custom_levels is not None:
            levels = [str(x) for x in custom_levels]
        else:
            levels = [str(lo_f), str(hi_f)]
        if len(levels) < 2:
            raise TypeError(
                f"@jev: {return_model.__name__}.{name}: a float score needs at least 2 levels"
            )
        n_levels = len(levels)

        def extract_float(r: _AnswersView) -> float:
            expected = r.scores[name].score  # in [0, n_levels - 1]
            return lo_f + expected * (hi_f - lo_f) / (n_levels - 1)

        extractor = extract_float

    if len(levels) > _MAX_SCORE_LEVELS:
        raise TypeError(
            f"@jev: {return_model.__name__}.{name} has {len(levels)} levels; "
            f"Jev supports at most {_MAX_SCORE_LEVELS} per score"
        )
    return Score(instructions=instructions, criteria=levels), extractor


# ---------------------------------------------------------------------------
# Call-time rendering and coercion
# ---------------------------------------------------------------------------


def _render_framing(
    template: jinja2.Template | None,
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    if template is None:
        return None
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    try:
        return template.render(**bound.arguments)
    except jinja2.UndefinedError as exc:
        raise TypeError(f"@jev: docstring template references an unknown variable: {exc}") from exc


def _body_less_state(
    template: jinja2.Template | None,
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """State for a body that did not build one: the rendered docstring, or the
    arguments themselves as JSON when there is no docstring."""
    framing = _render_framing(template, signature, args, kwargs)
    if framing is not None:
        return framing
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # Round-trip through JSON so the state is exactly what the wire would
    # carry: JSON-safe values only, anything exotic stringified.
    return json.loads(json.dumps(bound.arguments, default=str))


def _resolve_body(
    func: Callable[..., Any],
    return_model: type[BaseModel],
    template: jinja2.Template | None,
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    body_result: Any,
) -> tuple[BaseModel | None, Any]:
    """Interpret the evaluated body's result.

    Returns ``(override, None)`` when the body answered directly (skip the API
    call), or ``(None, state)`` to query Jev with the resolved state.
    """
    value = getattr(body_result, _STATE_ATTR, _ABSENT)
    if value is not _ABSENT:
        if type(body_result) is not return_model:
            raise TypeError(
                f"@jev: {func.__qualname__} returned a state marker built for "
                f"{type(body_result).__name__}, but its return annotation is "
                f"{return_model.__name__}; use {func.__qualname__}.state(...)"
            )
        if value is None:
            # fn.state() with no value: the body-less form.
            return None, _body_less_state(template, signature, args, kwargs)
        # fn.state(value): the value is the state, exactly as returned. The
        # docstring is documentation in this form and is not sent.
        return None, value

    if isinstance(body_result, return_model):
        # A real, fully-constructed model: the body answered directly.
        return body_result, None

    if body_result is not None:
        raise TypeError(
            f"@jev: {func.__qualname__}'s body must return "
            f"{func.__qualname__}.state(...) or nothing, "
            f"got {type(body_result).__name__}"
        )

    return None, _body_less_state(template, signature, args, kwargs)


def _call_body(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run the body; a bare ``raise NotImplementedError`` means body-less."""
    try:
        return func(*args, **kwargs)
    except NotImplementedError:
        return None


async def _acall_body(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    try:
        return await func(*args, **kwargs)
    except NotImplementedError:
        return None


def _rebuild_question(question: Noul | Choice | Score, index: int) -> Noul | Choice | Score:
    """The same question, addressed to the item at ``index`` in a state array."""
    instructions = f"For the item at index {index} in the state array: {question.instructions}"
    if isinstance(question, Noul):
        return Noul(instructions=instructions, criteria=question.criteria)
    if isinstance(question, Choice):
        return Choice(instructions=instructions, criteria=question.criteria)
    return Score(instructions=instructions, criteria=question.criteria)


def _bind_single(
    func: Callable[..., Any], signature: inspect.Signature, item: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        bound = signature.bind(item)
    except TypeError as exc:
        raise TypeError(
            f"@jev: {func.__qualname__}.map(items) needs each item to be the "
            f"function's only positional argument: {exc}"
        ) from exc
    bound.apply_defaults()
    return bound.args, bound.kwargs


# overrides by index, states by index, batched questions, and the state
# array (None in slots answered directly, so indices hold).
_MapPlan = tuple[
    dict[int, BaseModel], dict[int, Any], dict[str, Noul | Choice | Score], list[Any]
]


def _map_resolve(
    func: Callable[..., Any],
    return_model: type[BaseModel],
    template: jinja2.Template | None,
    signature: inspect.Signature,
    questions: dict[str, Noul | Choice | Score],
    bound: list[tuple[tuple[Any, ...], dict[str, Any]]],
    body_results: list[Any],
) -> _MapPlan:
    """Interpret each item's body result through the same machinery as a
    direct call, then batch the questions of the items that need Jev."""
    overrides: dict[int, BaseModel] = {}
    states: dict[int, Any] = {}
    for i, ((args, kwargs), body_result) in enumerate(zip(bound, body_results, strict=True)):
        override, state = _resolve_body(
            func, return_model, template, signature, args, kwargs, body_result
        )
        if override is not None:
            overrides[i] = override
        else:
            states[i] = state
    batched: dict[str, Noul | Choice | Score] = {}
    for i in states:
        for name, question in questions.items():
            batched[f"{i}:{name}"] = _rebuild_question(question, i)
    return overrides, states, batched, [states.get(i) for i in range(len(bound))]


def _map_prepare(
    func: Callable[..., Any],
    return_model: type[BaseModel],
    template: jinja2.Template | None,
    signature: inspect.Signature,
    questions: dict[str, Noul | Choice | Score],
    items: Sequence[JSONContent],
) -> _MapPlan:
    """Bind and run each item through the body, then resolve the plan."""
    bound = [_bind_single(func, signature, item) for item in items]
    body_results = [_call_body(func, args, kwargs) for args, kwargs in bound]
    return _map_resolve(
        func, return_model, template, signature, questions, bound, body_results
    )


async def _map_prepare_async(
    func: Callable[..., Any],
    return_model: type[BaseModel],
    template: jinja2.Template | None,
    signature: inspect.Signature,
    questions: dict[str, Noul | Choice | Score],
    items: Sequence[JSONContent],
) -> _MapPlan:
    bound = [_bind_single(func, signature, item) for item in items]
    body_results = [await _acall_body(func, args, kwargs) for args, kwargs in bound]
    return _map_resolve(
        func, return_model, template, signature, questions, bound, body_results
    )


def _map_finish(
    return_model: type[BaseModel],
    extractors: dict[str, _Extractor],
    response: SystemOneResponse | None,
    overrides: dict[int, BaseModel],
    states: dict[int, Any],
    n: int,
) -> list[Any]:
    results: list[Any] = [None] * n
    for i, override in overrides.items():
        results[i] = override
    if states:
        if response is None:
            raise TypeError("@jev: internal error: batched answers missing")
        for i in states:
            results[i] = return_model(**_extract_values(extractors, _AnswersView.for_item(response, i)))
    return results


def _extract_values(
    extractors: dict[str, _Extractor],
    response: _AnswersView,
) -> dict[str, Any]:
    return {name: extract(response) for name, extract in extractors.items()}


def _materialize(
    return_model: type[BaseModel],
    extractors: dict[str, _Extractor],
    response: SystemOneResponse,
) -> BaseModel:
    return return_model(**_extract_values(extractors, _AnswersView.whole(response)))


# ---------------------------------------------------------------------------
# Lazily-created shared clients (TYPESAFE_API_KEY / SDK defaults)
# ---------------------------------------------------------------------------

_shared_sync_client: TypeSafeClient | None = None
# httpx connection pools are bound to the event loop that created them, so a
# client must never outlive its loop: keep one client per running loop.
_shared_async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncTypeSafeClient] = (
    weakref.WeakKeyDictionary()
)


def _default_sync_client() -> TypeSafeClient:
    global _shared_sync_client
    if _shared_sync_client is None:
        _shared_sync_client = TypeSafeClient()
    return _shared_sync_client


def _default_async_client() -> AsyncTypeSafeClient:
    loop = asyncio.get_running_loop()
    client = _shared_async_clients.get(loop)
    if client is None:
        client = AsyncTypeSafeClient()
        _shared_async_clients[loop] = client
    return client


# ---------------------------------------------------------------------------
# JevModel: construct a model instance straight from a state
# ---------------------------------------------------------------------------


class JevModel(BaseModel):
    """A pydantic model whose fields are Jev questions.

    ``Model.decide(state)`` queries Jev and fills the fields from the answers;
    the normal pydantic constructor validates locally and skips the API (the
    mock seam). Fields follow the same rules as ``@jev`` return models: bool,
    Literal[...], Enum, or int/float with Field(ge=..., le=...). Field
    compilation happens at class definition, so an unsupported field type
    raises TypeError at import::

        class Triage(JevModel):
            department: Literal["billing", "technical", "sales"]
            is_urgent: bool

        Triage.decide("I was charged twice!")         # queries Jev
        await Triage.adecide("I was charged twice!")  # async form
        Triage(department="billing", is_urgent=True)  # no API call

    Class attributes: ``__jev_model__`` pins the model name,
    ``__jev_bool_threshold__`` overrides the Noul -> bool threshold.

    (Deciding is a classmethod rather than a constructor overload because
    pydantic's ``dataclass_transform`` synthesizes a field-only ``__init__``
    for subclasses in both mypy and pyright; a classmethod keeps the call
    typed as ``-> Self`` in both checkers.)
    """

    __jev_questions__: ClassVar[dict[str, Noul | Choice | Score]] = {}
    __jev_extractors__: ClassVar[dict[str, _Extractor]] = {}
    __jev_bool_threshold__: ClassVar[float | None] = None
    __jev_model__: ClassVar[str | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__jev_bool_threshold__ is not None:
            _validate_bool_threshold(
                cls.__jev_bool_threshold__, f"{cls.__name__}.__jev_bool_threshold__"
            )
        questions, extractors = _compile_questions(cls, cls.__jev_bool_threshold__)
        cls.__jev_questions__ = questions
        cls.__jev_extractors__ = extractors

    @classmethod
    def decide(cls, state: JSONContent) -> Self:
        """Decide the fields about a state by querying Jev."""
        if not cls.__jev_questions__:
            raise TypeError(f"{cls.__name__} declares no question fields")
        response = _default_sync_client().system_one(
            state=state, questions=cls.__jev_questions__, model=cls.__jev_model__
        )
        return cls(**_extract_values(cls.__jev_extractors__, _AnswersView.whole(response)))

    @classmethod
    async def adecide(cls, state: JSONContent) -> Self:
        """The async form of ``decide``."""
        if not cls.__jev_questions__:
            raise TypeError(f"{cls.__name__} declares no question fields")
        response = await _default_async_client().system_one(
            state=state, questions=cls.__jev_questions__, model=cls.__jev_model__
        )
        return cls(**_extract_values(cls.__jev_extractors__, _AnswersView.whole(response)))
