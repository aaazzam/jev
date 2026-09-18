# jev

`@jev.fn` turns a Python function definition into a query against [Jev](https://typesafe.ai), TypeSafe's System One model. You declare the function (parameters, docstring, return annotation) and the decorator compiles it into a `state` + typed `questions` request. Calling the function sends the request and returns a validated instance of the return annotation.

Jev generates no text. It answers typed questions about a state (yes/no probabilities, choices, scores) in one parallel call, with calibrated probabilities. That constraint drives the design: the return annotation must be a Pydantic model, and each of its fields maps onto one of Jev's three question types.

A function signature is already a complete specification of a decision. The name says what to decide, the parameters say what to decide it from, and the return annotation says what shape the answer takes; the docstring supplies the judgment. `@jev.fn` treats that specification as sufficient and lets Jev fill in the body. The spec is made of things you already write: a signature, a docstring, a Pydantic model. There is no prompt string to maintain, no JSON schema to keep in sync, no parsing layer between the call and the answer.

## Install

```sh
uv sync
```

Python 3.14. Get an API key from [console.typesafe.ai](https://console.typesafe.ai) and put it in `.env`:

```sh
TYPESAFE_API_KEY=...
```

## The pattern

```python
from typing import Literal
from pydantic import BaseModel, Field
import jev

class Triage(BaseModel):
    department: Literal["billing", "technical", "sales"]
    is_urgent: bool
    frustration: int = Field(ge=0, le=2)

@jev.fn
def triage(ticket: str) -> Triage:
    """A customer support ticket:

    {{ ticket }}
    """
    return triage.state()

triage("I was charged twice. Fix this NOW.")
# Triage(department='billing', is_urgent=True, frustration=2)
```

What happens:

1. At decoration time, the return annotation is checked (must be a `BaseModel` subclass; anything else raises `TypeError` at import rather than at first call) and each field is compiled into a question.
2. At call time, the docstring is rendered as a Jinja2 template with the bound arguments and sent as the `state`. With no docstring, the arguments themselves are sent as a JSON state.
3. The answers are coerced back and validated by pydantic. For the body-less form, write the body as `return triage.state()`; it type-checks like any other return and leaves the docstring as the whole state. A bare `...` or `raise NotImplementedError` also works.

For the body-less form the docstring does all the work, which may make it the only place in Python where documentation outranks implementation.

## Field mapping

| Field type | Jev question | Coerced back as |
|---|---|---|
| `bool` | Noul | `p(yes) >= threshold` (default 0.5) |
| `Literal[...]` | Choice | the selected label |
| `Enum` | Choice | the selected member |
| `int` with `Field(ge=, le=)` | Score | `lo + round(expected_score)` |
| `float` with `Field(ge=, le=)` | Score | linear interpolation over the levels |

- `Field(description=...)` becomes the question's instructions; without one the field name is humanized (`is_urgent` → "is urgent"). Write descriptions; they are the questions.
- Score levels default to the numbers in range. Override them with `Field(..., json_schema_extra={"levels": ["cold", "warm", "hot"]})`.
- Limits: 255 options per choice, 256 levels per score. Exceeding either is a `TypeError` at decoration time.
- Anything else (`str`, nested models, lists, `Optional`) raises `TypeError` at decoration time, because Jev cannot produce those values.

## Evaluated bodies

The body always runs, and there are three useful things it can do:

- **Return `fn.state()` with no value**: the body-less form; the rendered docstring is the whole state. (`...` or `raise NotImplementedError` work too.)
- **Build the state** with `return fn.state(value)`: loops, conditionals, f-strings, whatever Python you need; the returned value is the state, verbatim, and the docstring is documentation in this form.
- **Answer directly** with `return Model(...)`: skips the API call entirely, the mock seam for tests.

Building the state looks like this:

```python
@jev.fn
def triage_batch(tickets: list[str]) -> BatchTriage:
    """Triage a batch of support tickets."""
    numbered = [f"[{i}] {t}" for i, t in enumerate(tickets)]
    return triage_batch.state(
        "Triage this batch of support tickets.\n\n" + "\n".join(numbered)
    )
```

Framing like "Triage this batch" lives in the body now, in the open, rather than being lifted out of the docstring.

`fn.state(...)` is typed `value -> return-annotation`, so the body's `return` type-checks against the annotation.

Async functions work identically (`await` the call; the body may also `await`). Configure per function with `@jev.fn(model="jev-latest", client=...)`; the default client reads `TYPESAFE_API_KEY` and calls `jev-latest`. The bool threshold defaults to 0.5; tune it per function with `@jev.fn(bool_threshold=0.7)` or globally with the `JEV_BOOL_THRESHOLD` environment variable.

## Batch with `.map`

`triage.map(tickets)` applies the function to each item in ONE call: the items become a JSON state array, the fields become one set of questions per item, and Jev answers all of them in parallel. The result is a `list[Triage]` in input order. Each item runs through the same body machinery as a direct call, so short-circuited items (`return Model(...)`) skip the API and slot into the results, and evaluated bodies run per item. Items must bind as the function's only positional argument. Batches are a single request: hundreds of questions per call have worked in TypeSafe's own cookbooks, but there is no documented limit, so chunk very large batches yourself.

```python
triage.map(tickets)          # sync:  list[Triage]
await atriage.map(tickets)   # async: list[Triage]
```

## Class form: `jev.BaseModel`

For the common case of one blob of state in, one struct out, there is a class interface (the same field machinery, no docstring involved):

```python
import jev

class Triage(jev.BaseModel):
    department: Literal["billing", "technical", "sales"]
    is_urgent: bool
    frustration: int = Field(ge=0, le=2)

Triage.decide("I was charged twice. Fix this NOW.")
# Triage(department='billing', is_urgent=True, frustration=2)

await Triage.adecide("...")                        # async form
Triage(department="billing", is_urgent=False, frustration=0)  # plain constructor: no API call
```

Fields compile at class definition (unsupported types raise at import). Class attributes `__jev_model__` and `__jev_bool_threshold__` pin the model and threshold per class. Deciding is a classmethod rather than a constructor overload because pydantic's `dataclass_transform` synthesizes a field-only `__init__` for subclasses in both mypy and pyright; the classmethod keeps the call typed as `-> Self` in both.

## Function form: `jev.decide`

The same decision on a plain `BaseModel` — no subclass, no decorator:

```python
from pydantic import BaseModel
import jev

class Triage(BaseModel):
    department: Literal["billing", "technical", "sales"]
    is_urgent: bool
    frustration: int = Field(ge=0, le=2)

jev.decide("I was charged twice. Fix this NOW.", Triage)
# Triage(department='billing', is_urgent=True, frustration=2)

await jev.adecide("...", Triage)  # async form
```

Keyword arguments `model=` and `bool_threshold=` mirror the class attributes. Handed a `jev.BaseModel` subclass, `jev.decide` falls back to `__jev_model__` / `__jev_bool_threshold__` and reuses the questions compiled at class definition; plain models compile per call, so prefer the class form in hot loops.

## Testing

```python
from jev import builder, state_payload

marker = builder(triage_batch)(["a", "b"])   # runs the body, no API call
assert state_payload(marker) == "Triage this batch of support tickets.\n\n[0] a\n[1] b"
```

Or return a model from the body to short-circuit the call in tests.

## Type checking

The decorated function has type `JevFn[P, R]` (or `AsyncJevFn[P, R]` for async), so call sites see the original parameter signature and the declared return model, and `fn.state` returns the same model. Bare `@jev.fn` rejects a non-`BaseModel` return annotation statically, before any code runs. The package passes `pyright --strict` and `mypy --strict` with no casts and no ignore comments.

See `example.py` for a runnable tour (`uv run python example.py`).

## Limitations

- **Probabilities are discarded.** `bool` is thresholded (0.5 by default, tunable), Choice takes the argmax, Score returns the expected value. Jev's calibrated probabilities and confidence scores never reach you; if you need them (confidence-gated routing is the main reason to use Jev), use `typesafe_sdk` directly.
- **No streaming.** Jev samples in parallel in a single shot, so there is nothing to stream.
- **Legacy sentinel.** `raise NotImplementedError` (or `...`) still marks a body-less function, so a body that raises it incidentally is silently treated as body-less. Prefer `return fn.state()`, which carries no such ambiguity.
- **Clients live for the process lifetime.** The default sync client is created lazily and never closed; async clients are created per event loop (connection pools are loop-bound) and never closed.
- **The docstring is a template rather than documentation.** `help(fn)` shows Jinja. If that bothers you, keep the docstring minimal and do the work in an evaluated body.
