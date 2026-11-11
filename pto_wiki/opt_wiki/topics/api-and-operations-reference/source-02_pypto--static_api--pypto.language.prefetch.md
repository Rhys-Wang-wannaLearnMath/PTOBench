# `pypto.language.prefetch` — installed API reference

Extracted from `.venv-official` at PyPTO `747e1e4ba99f13dea1a609d893ac7512e85a5897`. Signatures and English docstrings; existence does not certify every backend/dtype/shape.

## `async_prefetch`

```text
pypto.language.prefetch.async_prefetch(src: pypto.language.typing.tensor.Tensor, ctx: pypto.language.typing.prefetch_handle.PrefetchAsyncContext) -> pypto.language.typing.prefetch_handle.AsyncEvent
```

Start one asynchronous prefetch of a GM region into L2 cache.

Does not block and does not modify ``src``.

Args:
    src: A flat contiguous logical-1D GM [`pl.Tensor`][pypto.language.Tensor] to pull into L2.
        The op verifier (C++) requires a fully static shape whose dimensions
        are all 1 except the last — e.g. ``[N]`` or ``[1, N]``.
    ctx: A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] from
        [`make_context`][pypto.language.prefetch.make_context].

Returns:
    An [`AsyncEvent`][pypto.language.AsyncEvent] to pass to [`wait`][pypto.language.prefetch.wait] along
    with the session.

## `make_context`

```text
pypto.language.prefetch.make_context() -> pypto.language.typing.prefetch_handle.PrefetchAsyncContext
```

Build an asynchronous-prefetch context with a runtime-injected workspace.

The caller supplies no workspace. Codegen and the runtime bind the returned
context to a hidden runtime-owned SDMA allocation.

Returns:
    A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] handle to pass to
    [`async_prefetch`][pypto.language.prefetch.async_prefetch]
    and [`session`][pypto.language.prefetch.session].

## `session`

```text
pypto.language.prefetch.session(ctx: pypto.language.typing.prefetch_handle.PrefetchAsyncContext) -> pypto.language.typing.prefetch_handle.AsyncSession
```

Project the asynchronous session bound to a prefetch context.

Args:
    ctx: A [`PrefetchAsyncContext`][pypto.language.PrefetchAsyncContext] from
        [`make_context`][pypto.language.prefetch.make_context].

Returns:
    An [`AsyncSession`][pypto.language.AsyncSession] to pass to [`wait`][pypto.language.prefetch.wait].

## `wait`

```text
pypto.language.prefetch.wait(event: pypto.language.typing.prefetch_handle.AsyncEvent, session_handle: pypto.language.typing.prefetch_handle.AsyncSession) -> pypto.language.typing.scalar.Scalar
```

Wait for an asynchronous prefetch event to complete within its session.

Call this before the hot loop that consumes the prefetched region so the
data is resident in L2.

Args:
    event: An [`AsyncEvent`][pypto.language.AsyncEvent] from
        [`async_prefetch`][pypto.language.prefetch.async_prefetch].
    session_handle: The matching [`AsyncSession`][pypto.language.AsyncSession] from
        [`session`][pypto.language.prefetch.session].

Returns:
    A ``BOOL`` [`Scalar`][pypto.language.Scalar] done flag.
