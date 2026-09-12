"""Can numba reach the instructions Stockfish uses for the output layer?

numba has no intrinsics, but @intrinsic is documented as "an escape hatch for expert users to
build custom LLVM IR that will be inlined into the caller", and that is enough to write vector
IR by hand. Two questions this answers, and only on a machine with AVX-512:

  1. Does explicit <32 x i16> IR get lowered to 512-bit registers at all?
  2. Does LLVM recognise sext-mul-add as vpmaddwd, or must the target intrinsic be named?

Correctness is checked against numpy on every run: hand-written IR that computes the wrong dot
product would be a silent evaluation bug, which is the worst kind this engine can have.
"""
import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

LANES = 32  # int16 lanes in a 512-bit register


@intrinsic
def dot_i16(typingctx, a_ptr, b_ptr, n):
    """sum(a[i] * b[i]) over n int16 pairs, widened to int32, as explicit vector IR."""
    if not isinstance(a_ptr, types.Array):
        return None
    sig = types.int32(a_ptr, b_ptr, types.int64)

    def codegen(context, builder, signature, args):
        a, b, count = args
        a_arr = cgutils.create_struct_proxy(signature.args[0])(context, builder, value=a)
        b_arr = cgutils.create_struct_proxy(signature.args[1])(context, builder, value=b)
        i16 = ir.IntType(16)
        i32 = ir.IntType(32)
        vec16 = ir.VectorType(i16, LANES)
        vec32 = ir.VectorType(i32, LANES)
        # A stack slot, not a value: a value defined inside the loop does not dominate its use
        # after it, which is an SSA violation LLVM rejects outright. mem2reg turns this back
        # into a phi node during optimisation, so it costs nothing.
        slot = cgutils.alloca_once(builder, vec32)
        builder.store(ir.Constant(vec32, None), slot)

        blocks = builder.sdiv(count, ir.Constant(ir.IntType(64), LANES))
        with cgutils.for_range(builder, blocks) as loop:
            offset = builder.mul(loop.index, ir.Constant(ir.IntType(64), LANES))
            pa = builder.gep(a_arr.data, [offset])
            pb = builder.gep(b_arr.data, [offset])
            va = builder.load(builder.bitcast(pa, vec16.as_pointer()))
            vb = builder.load(builder.bitcast(pb, vec16.as_pointer()))
            prod = builder.mul(builder.sext(va, vec32), builder.sext(vb, vec32))
            builder.store(builder.add(builder.load(slot), prod), slot)

        total = builder.load(slot)
        acc = ir.Constant(i32, 0)
        for lane in range(LANES):
            acc = builder.add(acc, builder.extract_element(total, ir.Constant(i32, lane)))
        return acc

    return sig, codegen


@intrinsic
def dot_i16_named(typingctx, a_ptr, b_ptr, n):
    """The same dot product, but calling llvm.x86.avx512.pmaddw.d.512 by name.

    pmaddwd multiplies int16 lanes pairwise and adds adjacent pairs into int32, which is one
    instruction for what the generic IR above expresses as sext, mul and add. Naming the
    intrinsic removes any question of whether the pattern matcher finds it -- at the cost of
    being x86-and-AVX-512 only, so it needs a runtime guard before it could ever ship.
    """
    if not isinstance(a_ptr, types.Array):
        return None
    sig = types.int32(a_ptr, b_ptr, types.int64)

    def codegen(context, builder, signature, args):
        a, b, count = args
        a_arr = cgutils.create_struct_proxy(signature.args[0])(context, builder, value=a)
        b_arr = cgutils.create_struct_proxy(signature.args[1])(context, builder, value=b)
        i16, i32 = ir.IntType(16), ir.IntType(32)
        vec16, vec32 = ir.VectorType(i16, LANES), ir.VectorType(i32, LANES // 2)
        module = builder.module
        fnty = ir.FunctionType(vec32, [vec16, vec16])
        try:
            pmaddwd = module.globals["llvm.x86.avx512.pmaddw.d.512"]
        except KeyError:
            pmaddwd = ir.Function(module, fnty, "llvm.x86.avx512.pmaddw.d.512")

        slot = cgutils.alloca_once(builder, vec32)
        builder.store(ir.Constant(vec32, None), slot)
        blocks = builder.sdiv(count, ir.Constant(ir.IntType(64), LANES))
        with cgutils.for_range(builder, blocks) as loop:
            offset = builder.mul(loop.index, ir.Constant(ir.IntType(64), LANES))
            va = builder.load(builder.bitcast(
                builder.gep(a_arr.data, [offset]), vec16.as_pointer()))
            vb = builder.load(builder.bitcast(
                builder.gep(b_arr.data, [offset]), vec16.as_pointer()))
            builder.store(
                builder.add(builder.load(slot), builder.call(pmaddwd, [va, vb])), slot)

        total = builder.load(slot)
        acc = ir.Constant(i32, 0)
        for lane in range(LANES // 2):
            acc = builder.add(acc, builder.extract_element(total, ir.Constant(i32, lane)))
        return acc

    return sig, codegen


@njit(cache=False)
def use(a, b):
    return dot_i16(a, b, a.size)


@njit(cache=False)
def use_named(a, b):
    return dot_i16_named(a, b, a.size)


def time_one(fn, a, b, a_type, b_type, rounds: int = 200_000) -> float:
    """Timed through a jitted driver: calling a numba function from Python costs about 230ns of
    dispatch, several times the thing being measured."""
    import time

    from numba import int64

    @njit(int64(int64, a_type, b_type), nogil=True, cache=False)
    def loop(n, x, y):
        total = 0
        for _ in range(n):
            total += fn(x, y)
        return total

    loop(1000, a, b)
    start = time.perf_counter_ns()
    loop(rounds, a, b)
    return (time.perf_counter_ns() - start) / rounds


@njit(cache=False)
def plain(a, b):
    """The formulation nnue.py ships, for scale."""
    total = np.int32(0)
    for i in range(a.size):
        total += np.int32(a[i]) * np.int32(b[i])
    return total


def main() -> None:
    from numba import typeof

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, 512).astype(np.int16)
    b = rng.integers(-127, 128, 512).astype(np.int16)
    want = int((a.astype(np.int32) * b.astype(np.int32)).sum())
    a_type, b_type = typeof(a), typeof(b)

    print(f"host cpu {llvm.get_host_cpu_name()}, {a.size} int16 pairs")
    for label, fn in (("plain numba loop", plain), ("generic vector IR", use),
                      ("named pmaddwd", use_named)):
        try:
            got = fn(a, b)
        except Exception as error:
            print(f"  {label:<18} FAILED: {type(error).__name__}: {str(error)[:80]}")
            continue
        asm = "\n".join(fn.inspect_asm().values())
        counts = " ".join(f"{op} {asm.count(op)}" for op in ("vpmaddwd", "vpmulld", "zmm"))
        best = min(time_one(fn, a, b, a_type, b_type) for _ in range(5))
        verdict = "MATCH" if got == want else f"MISMATCH {got} vs {want}"
        print(f"  {label:<18} {verdict:<8} {best:6.1f}ns   {counts}")


if __name__ == "__main__":
    main()
