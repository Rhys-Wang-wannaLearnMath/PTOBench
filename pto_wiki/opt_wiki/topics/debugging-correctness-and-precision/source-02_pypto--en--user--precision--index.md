> **Local reading edition:** 910B4 / CANN 9.0.0 / pinned PyPTO 747e1e4b / PTOAS 0.60. See [environment and applicability](../setup-and-first-kernels/source-BASELINE.md).

# Precision

The result is wrong, not slow. This chapter is the order in which to suspect things.

> **Prerequisites:** [Execution](../execution-and-runtime/source-02_pypto--en--user--execution--index.md) — compiling, running, and the
> `RunConfig` fields this chapter turns on. Also [Tutorials](../kernel-authoring-and-tutorials/source-02_pypto--en--user--tutorials--index.md) — specifically the `allclose`
> comparison from [Your first operator](../kernel-authoring-and-tutorials/source-02_pypto--en--user--tutorials--00-elementwise.md).

## What this chapter is

Not "how to use an API". A **narrowing procedure**: five steps, ordered so the two that cost
minutes come before the two that cost hours.

Most reports stop at step 2, because a large share of "wrong results" are either a
mis-specified tolerance or a difference that is supposed to be there.

## Contents

| Page | Covers |
| ---- | ------ |
| [Narrowing down a gap](source-02_pypto--en--user--precision--00-workflow.md) | The five steps, the tools, and the acceptable-difference table |
| [Worked cases](source-02_pypto--en--user--precision--01-cases.md) | The procedure applied end to end |

## The shape of it

```text
1. Is the golden right?        ← minutes
2. Should it differ at all?    ← minutes
3. Did the compiler warn?      ← minutes
4. Which pass introduced it?   ← hours
5. Which tensor is wrong?      ← hours
```

**The first three are cheap and disqualifying.** A correct FP16 kernel fails a `1e-5`
tolerance; split-K reorders its accumulation by design. Neither is a bug, and both look
exactly like one from the outside.

## The distinction that matters most

There are two failure modes and they need different tools:

| Failure mode | Symptom | Located by |
| ------------ | ------- | ---------- |
| **Semantic** | The IR itself computes the wrong thing | `torch_codegen` + `validate_ir`, bisecting passes (step 4) |
| **Data** | The IR is right at every pass; the device result is not | Tensor dumps (step 5) |

Step 4 runs the IR's meaning **on the host**, no device involved. If it matches at every
pass, the bug is not semantic and no amount of IR reading will find it — go to step 5. That
one branch saves most of the time people lose here.

## See Also

- [Performance](../performance-and-optimization/source-02_pypto--en--user--performance--index.md) — the sibling loop, for slow rather than wrong.
- [LegalizeTileCast](../compiler-and-ir-internals/source-02_pypto--en--dev--passes--17-legalize_tile_cast.md) — when a cast chain is
  exact, and when it is not.
- [Reduction and softmax](../kernel-authoring-and-tutorials/source-02_pypto--en--user--tutorials--01-reduction-softmax.md) — padding in reductions, a
  frequent source of quiet wrongness.
