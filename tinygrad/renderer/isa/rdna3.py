# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import struct, math, ctypes, types
from functools import cache, partial
from enum import Enum
from typing import cast, Callable
from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, PreRegAllocContext, imm
from tinygrad.helpers import unwrap, Target, lo32, hi32, round_up
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc

# ***** RDNA3 Ops *****

class RDNA3Ops(FastEnum):
  # SOPK
  S_MOVK_I32 = auto()
  # SOP1
  S_SEXT_I32_I8 = auto(); S_SEXT_I32_I16 = auto()
  S_MOV_B32 = auto()
  # SOP2
  S_ADD_U32 = auto(); S_SUB_U32 = auto()
  S_LSHL_B32 = auto(); S_LSHL_B64 = auto()
  S_LSHR_B32 = auto(); S_LSHR_B64 = auto()
  S_ASHR_I32 = auto(); S_ASHR_I64 = auto()
  S_MUL_I32 = auto()
  S_AND_B32 = auto(); S_AND_B64 = auto()
  S_XOR_B32 = auto(); S_XOR_B64 = auto()
  S_OR_B32 = auto(); S_OR_B64 = auto()
  S_CSELECT_B32 = auto(); S_CSELECT_B64 = auto()
  S_MAX_I32 = auto(); S_MAX_U32 = auto()
  S_MIN_I32 = auto(); S_MIN_U32 = auto()
  # SOPC
  S_CMP_LT_I32 = auto(); S_CMP_LT_U32 = auto()
  S_CMP_EQ_I32 = auto(); S_CMP_EQ_U32 = auto(); S_CMP_EQ_U64 = auto()
  S_CMP_LG_I32 = auto(); S_CMP_LG_U32 = auto(); S_CMP_LG_U64 = auto()
  # VOP1
  V_CVT_U32_U16 = auto(); V_CVT_I32_I16 = auto()
  V_CVT_U16_F16 = auto(); V_CVT_I16_F16 = auto()
  V_CVT_F16_U16 = auto(); V_CVT_F16_I16 = auto()
  V_CVT_U32_F32 = auto(); V_CVT_I32_F32 = auto()
  V_CVT_F32_U32 = auto(); V_CVT_F32_I32 = auto()
  V_CVT_I32_F64 = auto(); V_CVT_F64_I32 = auto()
  V_CVT_U32_F64 = auto(); V_CVT_F64_U32 = auto()
  V_CVT_F16_F32 = auto(); V_CVT_F32_F16 = auto()
  V_CVT_F32_F64 = auto(); V_CVT_F64_F32 = auto()
  # VOP2
  V_CNDMASK_B32 = auto()
  V_ADD_NC_U16 = auto(); V_ADD_NC_U32 = auto()
  V_SUB_NC_U16 = auto(); V_SUB_NC_U32 = auto(); V_SUBREV_NC_U32 = auto()
  V_SUBREV_F16 = auto(); V_SUBREV_F32 = auto()
  V_LSHLREV_B32 = auto(); V_LSHRREV_B32 = auto(); V_ASHRREV_I32 = auto()
  V_ADD_F16 = auto(); V_ADD_F32 = auto(); V_ADD_F64 = auto()
  V_SUB_F16 = auto(); V_SUB_F32 = auto(); V_SUB_F64 = auto()
  V_MUL_F16 = auto(); V_MUL_F32 = auto(); V_MUL_F64 = auto()
  V_MIN_F16 = auto(); V_MIN_F32 = auto(); V_MIN_I32 = auto(); V_MIN_U32 = auto()
  V_MAX_F16 = auto(); V_MAX_F32 = auto(); V_MAX_I32 = auto(); V_MAX_U32 = auto()
  V_AND_B32 = auto(); V_XOR_B32 = auto(); V_OR_B32 = auto()
  V_EXP_F16 = auto(); V_EXP_F32 = auto()
  V_LOG_F16 = auto(); V_LOG_F32 = auto()
  V_SIN_F16 = auto(); V_SIN_F32 = auto()
  V_RCP_F16 = auto(); V_RCP_F32 = auto(); V_RCP_F64 = auto()
  V_SQRT_F16 = auto(); V_SQRT_F32 = auto(); V_SQRT_F64 = auto()
  V_TRUNC_F16 = auto(); V_TRUNC_F32 = auto(); V_TRUNC_F64 = auto()
  # fmas
  V_FMAC_F16 = auto(); V_FMAC_F32 = auto()
  V_FMAAK_F16 = auto(); V_FMAAK_F32 = auto()
  V_FMAMK_F16 = auto(); V_FMAMK_F32 = auto()
  V_PK_FMAC_F16 = auto()
  # VOPC
  V_CMP_LT_F16 = auto(); V_CMP_LT_F32 = auto(); V_CMP_LT_F64 = auto()
  V_CMP_EQ_F16 = auto(); V_CMP_EQ_F32 = auto(); V_CMP_EQ_F64 = auto()
  V_CMP_NEQ_F16 = auto(); V_CMP_NEQ_F32 = auto(); V_CMP_NEQ_F64 = auto()
  V_CMP_LT_I16 = auto(); V_CMP_LT_I32 = auto(); V_CMP_LT_I64 = auto()
  V_CMP_LT_U16 = auto(); V_CMP_LT_U32 = auto(); V_CMP_LT_U64 = auto()
  V_CMP_EQ_I16 = auto(); V_CMP_EQ_I32 = auto(); V_CMP_EQ_I64 = auto()
  V_CMP_EQ_U16 = auto(); V_CMP_EQ_U32 = auto(); V_CMP_EQ_U64 = auto()
  V_CMP_NE_I16 = auto(); V_CMP_NE_I32 = auto(); V_CMP_NE_I64 = auto()
  V_CMP_NE_U16 = auto(); V_CMP_NE_U32 = auto(); V_CMP_NE_U64 = auto()
  # VOP3
  V_FMA_F16 = auto(); V_FMA_F32 = auto(); V_FMA_F64 = auto()
  V_CNDMASK_B16 = auto()
  V_AND_B16 = auto(); V_XOR_B16 = auto(); V_OR_B16 = auto()
  V_LSHLREV_B16 = auto(); V_LSHLREV_B64 = auto()
  V_LSHRREV_B16 = auto(); V_LSHRREV_B64 = auto()
  V_ASHRREV_I16 = auto(); V_ASHRREV_I64 = auto()
  V_MAX_I16 = auto(); V_MAX_U16 = auto(); V_MAX_F64 = auto()
  V_MIN_I16 = auto(); V_MIN_U16 = auto(); V_MIN_F64 = auto()
  # SMEM
  S_LOAD_B32 = auto(); S_LOAD_B64 = auto(); S_LOAD_B128 = auto(); S_LOAD_B256 = auto(); S_LOAD_B512 = auto()
  # LDS
  DS_LOAD_B32 = auto(); DS_LOAD_B64 = auto(); DS_LOAD_B96 = auto(); DS_LOAD_B128 = auto()
  DS_STORE_B8 = auto(); DS_STORE_B16 = auto(); DS_STORE_B32 = auto(); DS_STORE_B64 = auto(); DS_STORE_B96 = auto(); DS_STORE_B128 = auto()
  # GLOBAL
  GLOBAL_LOAD_B32 = auto(); GLOBAL_LOAD_B64 = auto(); GLOBAL_LOAD_B96 = auto(); GLOBAL_LOAD_B128 = auto()
  GLOBAL_STORE_B8 = auto(); GLOBAL_STORE_B16 = auto(); GLOBAL_STORE_B32 = auto(); GLOBAL_STORE_B64 = auto(); GLOBAL_STORE_B96 = auto(); GLOBAL_STORE_B128 = auto()

class RDNA3GroupOp:
  # RDNA3Ops whose first src is also the destination
  TwoAddress = {RDNA3Ops.V_FMAC_F16, RDNA3Ops.V_FMAC_F32}
  # VOP1/2 RDNA3Ops that don't have a VOP3 version
  NO_VOP3 = {RDNA3Ops.V_FMAAK_F16, RDNA3Ops.V_FMAAK_F32, RDNA3Ops.V_FMAMK_F16, RDNA3Ops.V_FMAMK_F32}

class Mod(Enum):
  FNEG = auto(); FABS = auto(); FNEG_FABS = auto()

# ***** RDNA3 legalization *****

extra_matcher = PatternMatcher([
  # TODO: this isn't actually required, just a noop
  (UPat.var("y", dtypes.bool).cast().named("x"), lambda y,x: y.where(x.const_like(1), x.const_like(0))),
])

# ***** RDNA3 registers *****

SCC = Register("scc", 0)
VCC = Register("vcc", 0)
EXEC = Register("exec", 0)

# scalar registers, the range step is the alignment required
S32 = tuple(Register(f"s{i}", i) for i in range(106))
S64 = tuple(Register(f"s[{i}:{i+2-1}]", i, S32[i:i+2]) for i in range(0, 106, 2))
S128 = tuple(Register(f"s[{i}:{i+4-1}]", i, S32[i:i+4]) for i in range(0, 106, 4))
S256 = tuple(Register(f"s[{i}:{i+8-1}]", i, S32[i:i+8]) for i in range(0, 106, 4))
S512 = tuple(Register(f"s[{i}:{i+32-1}]", i, S32[i:i+32]) for i in range(0, 106, 4))

SGPR = S32

# vector registers, the smallest unit is 16bit as instructions may specify access to the low or high 16bits of a vgpr
V16 = tuple(Register(f"v{i//2}.{'lh'[i%2]}", i) for i in range(512))
V32 = tuple(Register(f"v{i}", i, V16[i:i+2]) for i in range(256))
V64 = tuple(Register(f"v[{i}:{i+2-1}]", i, V32[i:i+2]) for i in range(256))
V128 = tuple(Register(f"v[{i}:{i+4-1}]", i, V32[i:i+4]) for i in range(256))
V16_COMPACT = V16[:256]

VGPR = V16

# ***** RDNA3 instruction selection *****

def is_inline_const(c:UOp) -> bool:
  if c.op is not Ops.CONST: return False
  if c.dtype in dtypes.ints: return -16 <= c.arg <= 64
  if c.dtype in dtypes.floats: return c.arg in {0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*math.pi)}

# a literal must fit in 32bits
def to_literal(c:UOp) -> UOp|None:
  if c.op is not Ops.CONST: return None
  if c.dtype is dtypes.int64: return imm(dtypes.int32, c.arg) if not c.overflows(dtypes.int32) else None
  if c.dtype is dtypes.uint64: return imm(dtypes.uint32, c.arg) if not c.overflows(dtypes.uint32) else None
  if c.dtype is dtypes.float64: return imm(dtypes.float64, c.arg) if lo32(struct.unpack('Q', struct.pack('d', c.arg))[0]) == 0 else None
  return imm(c.dtype, c.arg)

# pattern
#def partial(fxn, op):
#  defaults = (fxn.__defaults__ or ())[:-1] + (op,)   # swap the last default (op)
#  return types.FunctionType(fxn.__code__, fxn.__globals__, fxn.__name__, defaults, fxn.__closure__)

@cache
def is_uniform(x:UOp) -> bool:
  if x.op is Ops.SPECIAL and x.arg.startswith("lidx"): return False
  return all(is_uniform(s) for s in x.src)

def lower_sop1(ctx:IselContext, x:UOp, op:RDNA3Ops) -> UOp|None:
  if not is_uniform(x): return None
  return x.ins(op, tag=ctx.vreg(SGPR))

# SOP1
isel = [
  (UPat(dtype=dtypes.int8).cast(dtypes.int32, name="x"), partial(lower_sop1, op=RDNA3Ops.S_SEXT_I32_I8)),
  (UPat(dtype=dtypes.int16).cast(dtypes.int32, name="x"), partial(lower_sop1, op=RDNA3Ops.S_SEXT_I32_I16)),
]

# this is used when any src can be scalar, which is the case for SOP and VOP3 instructions
def ssrcs(src:tuple[UOp, ...]) -> tuple[UOp, ...]:
  # any src can be an inline constant, only one can be a literal
  ret, free_literal = [], True
  for s in src:
    if (sl:=to_literal(s)) is not None and (free_literal or is_inline_const(sl)):
      if not is_inline_const(sl): free_literal = False
      s = sl
    ret.append(s)
  return tuple(ret)

def lower_sop2(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp|None=None, b:UOp|None=None) -> UOp|None:
  if not is_uniform(x): return None
  return x.ins(op, src=ssrcs(x.src), tag=ctx.vreg(SGPR))

def lower_s_cselect(op:RDNA3Ops, ctx:IselContext, a:UOp, b:UOp, c:UOp, x:UOp) -> UOp|None:
  return lower_sop2(op, ctx, x.replace(src=(a, b, c)))

# SOP2
isel += [
  ((UPat(dtype=dtypes.int32s).alu(Ops.SUB, UPat())).named("x"), partial(lower_sop2, RDNA3Ops.S_SUB_U32)),
  ((UPat(dtype=dtypes.int32s) + UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_ADD_U32)),
  ((UPat(dtype=dtypes.int32s) * UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_MUL_I32)),
  ((UPat(dtype=dtypes.int32s) << UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHL_B32)),
  ((UPat(dtype=dtypes.int64s) << UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHL_B64)),
  ((UPat(dtype=dtypes.uint32) >> UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHR_B32)),
  ((UPat(dtype=dtypes.uint64) >> UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHR_B64)),
  ((UPat(dtype=dtypes.int32) >> UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_ASHR_I32)),
  ((UPat(dtype=dtypes.int64) >> UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_ASHR_I64)),
  ((UPat(dtype=dtypes.int32s) & UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_AND_B32)),
  ((UPat(dtype=dtypes.int64s) & UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_AND_B64)),
  ((UPat(dtype=dtypes.int32s) ^ UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_XOR_B32)),
  ((UPat(dtype=dtypes.int64s) ^ UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_XOR_B64)),
  ((UPat(dtype=dtypes.int32s) | UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_OR_B32)),
  ((UPat(dtype=dtypes.int64s) | UPat()).named("x"), partial(lower_sop2, RDNA3Ops.S_OR_B64)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.int32), UPat.var("a")), partial(lower_sop2, RDNA3Ops.S_MAX_I32)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.uint32), UPat.var("a")), partial(lower_sop2, RDNA3Ops.S_MAX_U32)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.int32), UPat.var("b")), partial(lower_sop2, RDNA3Ops.S_MIN_I32)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.uint32), UPat.var("b")), partial(lower_sop2, RDNA3Ops.S_MIN_U32)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("a", dtypes.int32s), UPat.var("b")).named("x"), partial(lower_s_cselect, RDNA3Ops.S_CSELECT_B32)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("a", dtypes.int64s), UPat.var("b")).named("x"), partial(lower_s_cselect, RDNA3Ops.S_CSELECT_B64)),
]

# NOTE: any SOP instruction can read SCC directly, S_CSELECT_B32/64 MUST read SCC
# do this in a separate pass that adds the extra required moves
def lower_sopc(op:RDNA3Ops, ctx:IselContext, x:UOp) -> UOp|None:
  if not is_uniform(x): return None
  # this is true when the consumer expects a mask so we convert
  if x.tag: return x.ins(RDNA3Ops.S_CSELECT_B32, src=(x.const_like(-1), x.const_like(0), x.rtag(None)), tag=ctx.vreg(VCC))
  return x.ins(op, src=ssrcs(x.src), tag=ctx.vreg(SCC))

# all sopc 64bit comparisons are sign agnostic unless a negative literal is present as
# literals in 64bit instructions are extended according to the sign of the instruction and sign ext != zero ext for negative ints
def lower_sopc_64bit(op:RDNA3Ops, ctx:IselContext, x:UOp) -> UOp|None:
  # an int >= -16 and <= 64 becomes an inline constant instead of a literal
  if x.dtype is dtypes.int64 and x.src[1].op is Ops.CONST and dtypes.int32.min <= x.src[1].arg < -16: return None
  return lower_sopc(ctx, x, op)

# SOPC
isel += [
  # comparisons whose user doesn't use the flag, move flag result to register
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LT_I32)),
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LT_U32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_EQ_I32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_EQ_U32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.int64s), UPat()), name="x"), partial(lower_sopc_64bit, RDNA3Ops.S_CMP_EQ_U64)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LG_I32)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LG_U32)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.int64s), UPat()), name="x"), partial(lower_sopc_64bit, RDNA3Ops.S_CMP_LG_U64)),
]

def lower_smem(ctx:IselContext, x:UOp) -> UOp|None:
  if not is_uniform(x): return None
  sz = x.dtype.itemsize
  op = {32: RDNA3Ops.S_LOAD_B32, 64: RDNA3Ops.S_LOAD_B32, 128: RDNA3Ops.S_LOAD_B128, 256: RDNA3Ops.S_LOAD_B256, 512: RDNA3Ops.S_LOAD_B512}[sz]
  return x.ins(op, tag=ctx.vreg(SGPR))

# SMEM
isel += [(UPat(Ops.LOAD, name="x"), lower_smem)]

def lower_vop1(op:RDNA3Ops, ctx:IselContext, x:UOp) -> UOp:
  if is_vop3(ctx, x): return lower_vop3(op, ctx, x)
  return x.ins(op, tag=ctx.vreg(VGPR))

# VOP1
isel += [
  # transcedental TODO: precision is probably wrong
  (UPat(Ops.EXP2, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_EXP_F16)),
  (UPat(Ops.EXP2, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_EXP_F32)),
  (UPat(Ops.LOG2, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_LOG_F16)),
  (UPat(Ops.LOG2, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_LOG_F32)),
  #TODO:
  #t = v_mul_f32  θ, 0x3e22f983     ; θ · (1/2π)  — 1/2π is an inline constant, free
  #r = v_fract_f32 t                ; reduce to [0,1) — exact, since period is 1.0
  #d = v_sin_f32  r                 ; sin(2π·r)
  (UPat(Ops.SIN, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_SIN_F16)),
  (UPat(Ops.SIN, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_SIN_F32)),
  (UPat(Ops.SQRT, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_SQRT_F16)),
  (UPat(Ops.SQRT, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_SQRT_F32)),
  (UPat(Ops.SQRT, dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_SQRT_F64)),
  (UPat(Ops.TRUNC, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_TRUNC_F16)),
  (UPat(Ops.TRUNC, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_TRUNC_F32)),
  (UPat(Ops.TRUNC, dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_TRUNC_F64)),
  (UPat(Ops.RECIPROCAL, dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_RCP_F16)),
  (UPat(Ops.RECIPROCAL, dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_RCP_F32)),
  (UPat(Ops.RECIPROCAL, dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_RCP_F64)),
  # int to int casts
  (UPat(dtype=dtypes.uint16).cast(dtypes.uint32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_U32_U16)),
  (UPat(dtype=dtypes.int16).cast(dtypes.int32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_I32_I16)),
  # int to float casts
  (UPat(dtype=dtypes.uint16).cast(dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F16_U16)),
  (UPat(dtype=dtypes.int16).cast(dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F16_I16)),
  (UPat(dtype=dtypes.uint32).cast(dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F32_U32)),
  (UPat(dtype=dtypes.int32).cast(dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F32_I32)),
  (UPat(dtype=dtypes.uint32).cast(dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F64_U32)),
  (UPat(dtype=dtypes.int32).cast(dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F64_I32)),
  # float to int casts
  (UPat(dtype=dtypes.float16).cast(dtypes.uint16, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_U16_F16)),
  (UPat(dtype=dtypes.float16).cast(dtypes.int16, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_I16_F16)),
  (UPat(dtype=dtypes.float32).cast(dtypes.uint32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_U32_F32)),
  (UPat(dtype=dtypes.float32).cast(dtypes.int32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_I32_F32)),
  (UPat(dtype=dtypes.float64).cast(dtypes.int32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_I32_F64)),
  (UPat(dtype=dtypes.float64).cast(dtypes.uint32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_U32_F64)),
  # float to float casts
  (UPat(dtype=dtypes.float32).cast(dtypes.float16, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F16_F32)),
  (UPat(dtype=dtypes.float16).cast(dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F32_F16)),
  (UPat(dtype=dtypes.float64).cast(dtypes.float32, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F32_F64)),
  (UPat(dtype=dtypes.float32).cast(dtypes.float64, name="x"), partial(lower_vop1, RDNA3Ops.V_CVT_F64_F32)),
]

# in VOP2 only src0 can be scalar (sgpr/inline/literal), otherwise the longer VOP3 encoding must be used
def vop2_src(a:UOp, b:UOp, cm:bool) -> tuple[UOp, UOp]:
  if (al:=to_literal(a)) is not None: return (al, b)
  # if src1 is a literal we try to commute
  if cm and (bl:=to_literal(b)) is not None: return (bl, a)
  # if src0 is divergent and src1 is uniform we try to commute in case src1 lands in sgpr
  if cm and not is_uniform(a) and is_uniform(b): return (b, a)
  return (a, b)

def lower_vop2(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp|None=None, b:UOp|None=None, cm:bool|None=None) -> UOp:
  if is_vop3(ctx, x): return lower_vop3(op, ctx, x)
  if a is None: a = x.src[0]
  if b is None: b = x.src[1]
  if cm is None: cm = x.op in GroupOp.Commutative
  return x.ins(op, src=vop2_src(a, b, cm), tag=ctx.vreg(VGPR))

# VOP2
isel += [
  ((UPat(dtype=dtypes.int32s) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_NC_U32)),
  ((UPat(dtype=dtypes.float16) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_F16)),
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_F32)),
  ((UPat(dtype=dtypes.float16) * UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_MUL_F16)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_MUL_F32)),
  (UPat.var("b", dtypes.int32s).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_NC_U32)),
  (UPat.var("b", dtypes.float16).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_F16)),
  (UPat.var("b", dtypes.float32).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_F32)),
  (UPat(dtype=dtypes.int32s).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_NC_U32)),
  (UPat(dtype=dtypes.float16).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_F16)),
  (UPat(dtype=dtypes.float32).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_F32)),
  ((UPat(dtype=dtypes.int32s) & UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_AND_B32)),
  ((UPat(dtype=dtypes.int32s) ^ UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_XOR_B32)),
  ((UPat(dtype=dtypes.int32s) | UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_OR_B32)),
  ((UPat.var("b", dtypes.int32s) << UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_LSHLREV_B32)),
  ((UPat.var("b", dtypes.uint32) >> UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_LSHRREV_B32)),
  ((UPat.var("b", dtypes.int32) >> UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_ASHRREV_I32)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float16), UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_MAX_F16, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float32), UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_MAX_F32, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.int32), UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_MAX_I32, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.uint32), UPat.var("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_MAX_U32, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float16), UPat.var("b")).named("x"), partial(lower_vop2, RDNA3Ops.V_MIN_F16, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float32), UPat.var("b")).named("x"), partial(lower_vop2, RDNA3Ops.V_MIN_F32, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.int32), UPat.var("b")).named("x"), partial(lower_vop2, RDNA3Ops.V_MIN_I32, cm=True)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.uint32), UPat.var("b")).named("x"), partial(lower_vop2, RDNA3Ops.V_MIN_U32, cm=True)),
]

def lower_vopc(op:RDNA3Ops, ctx:IselContext, x:UOp) -> UOp:
  if is_vop3(ctx, x): return lower_vop3(op, ctx, x)
  return x.ins(op, src=vop2_src(x.src[0], x.src[1], x.op in GroupOp.Commutative), tag=ctx.vreg(VCC))

# VOPC
isel += [
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_F16)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_F32)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_F64)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_F16)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_F32)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_F64)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NEQ_F16)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NEQ_F32)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.float64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NEQ_F64)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.int16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_I16)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_I32)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.int64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_I64)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uint16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_U16)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_U32)),
  (UPat(Ops.CMPLT, src=(UPat(dtype=dtypes.uint64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_LT_U64)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_I16)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_I32)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.int64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_I64)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.uint16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_U16)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_U32)),
  (UPat(Ops.CMPEQ, src=(UPat(dtype=dtypes.uint64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_EQ_U64)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.int16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_I16)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_I32)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.int64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_I64)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.uint16), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_U16)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_U32)),
  (UPat(Ops.CMPNE, src=(UPat(dtype=dtypes.uint64), UPat()), name="x"), partial(lower_vopc, RDNA3Ops.V_CMP_NE_U64)),
]

def is_vop3(ctx:IselContext, x:UOp):
  # TODO: include ABS, OMOD and CLMP modifiers
  return x.dtype in dtypes.floats and any(s.op is Ops.NEG and ctx.is_foldable(x, s) for s in x.src)

def lower_vop3(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp|None=None, b:UOp|None=None, c:UOp|None=None) -> UOp:
  if a is None: a = x.src[0]
  if b is None and len(x.src) >= 2: b = x.src[1]
  if c is None and len(x.src) == 3: c = x.src[2]
  src = tuple(s.replace(op=Ops.NOOP, arg=Ops.NEG) if s.op is Ops.NEG and ctx.is_foldable(x, s) else s for s in ssrcs(a, b, c))
  return x.ins(op, src=src, tag=ctx.vreg(VGPR))

def lower_v_cndmask(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp, b:UOp, c:UOp) -> UOp:
  ret = lower_vop3(op, ctx, x, a, b, c)
  return ret.replace(src=ret.src[:2] + (ret.src[2].rtag(),))

# VOP3
isel += [
  (UPat(dtype=dtypes.int16s).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_SUB_NC_U16)),
  ((UPat(dtype=dtypes.int16s) + UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_ADD_NC_U16)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_ADD_F64)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_MUL_F64)),
  ((UPat(dtype=dtypes.int16s) & UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_AND_B16)),
  ((UPat(dtype=dtypes.int16s) ^ UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_XOR_B16)),
  ((UPat(dtype=dtypes.int16s) | UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_OR_B16)),
  ((UPat.var("b", dtypes.int16s) << UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_LSHLREV_B16)),
  ((UPat.var("b", dtypes.int64s) << UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_LSHLREV_B64)),
  ((UPat.var("b", dtypes.uint16) >> UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_LSHRREV_B16)),
  ((UPat.var("b", dtypes.uint64) >> UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_LSHRREV_B64)),
  ((UPat.var("b", dtypes.int16) >> UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_ASHRREV_I16)),
  ((UPat.var("b", dtypes.int64) >> UPat.var("a")).named("x"), partial(lower_vop3, RDNA3Ops.V_ASHRREV_I64)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.int16), UPat.var("a")), partial(lower_vop3, RDNA3Ops.V_MAX_I16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.uint16), UPat.var("a")), partial(lower_vop3, RDNA3Ops.V_MAX_U16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float64), UPat.var("a")), partial(lower_vop3, RDNA3Ops.V_MAX_F64)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.int16), UPat.var("b")), partial(lower_vop3, RDNA3Ops.V_MIN_I16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.uint16), UPat.var("b")), partial(lower_vop3, RDNA3Ops.V_MIN_U16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float64), UPat.var("b")), partial(lower_vop3, RDNA3Ops.V_MIN_F64)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("b", dtypes.int16s+(dtypes.float16,)), UPat.var("a")).named("x"), partial(lower_v_cndmask, RDNA3Ops.V_CNDMASK_B16)),
  # NOTE: this is VOP2 but you want to lower as VOP3 as it can have 2 CONST srcs and isn't commutative
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("b", dtypes.int32s+(dtypes.float32,)), UPat.var("a")).named("x"), partial(lower_v_cndmask, RDNA3Ops.V_CNDMASK_B32)),
]

def lower_fma(ctx:IselContext, x:UOp, a:UOp, b:UOp):
  if not ctx.is_foldable(x, a): return None
  if x.dtype is dtypes.float16:
    if x.dtype.count == 2: return x.ins(RDNA3Ops.V_PK_FMAC_F16, src=(*a.src, b))
    if (i:=to_literal(b)) is not None: return x.ins(RDNA3Ops.V_FMAAK_F16, src=(*a.src, i))
    if (i:=to_literal(a.src[1])) is not None: return x.ins(RDNA3Ops.V_FMAMK_F16, src=(a.src[0], i, b))
    return x.ins(RDNA3Ops.V_FMAC_F16, src=(*a.src, b))
  if x.dtype is dtypes.float32:
    if (i:=to_literal(b)) is not None: return x.ins(RDNA3Ops.V_FMAAK_F32, src=(*a.src, i))
    if (i:=to_literal(a.src[1])) is not None: return x.ins(RDNA3Ops.V_FMAMK_F32, src=(a.src[0], i, b))
    return x.ins(RDNA3Ops.V_FMAC_F32, src=(*a.src, b))
  if x.dtype is dtypes.float64: return lower_vop3(ctx, x.replace(src=(*a.src, b)), RDNA3Ops.V_FMA_F64)
  return None

isel += [((UPat(Ops.MUL, dtypes.floats, name="a") + UPat.var("b")).named("x"), lower_fma)]

# rdna3 address spec is [base(sgpr pair) + soffset(sgpr) + offset]
def smem_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  base, idx = x.src
  scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1

  def _is_soffset(x:UOp): return x.vmin >= 0 and x.vmax * scale <= dtypes.int32.max
  def _is_offset(x:UOp): return x.op is Ops.CONST and -(1 << 20) <= x.arg * scale < (1 << 20)

  if idx.op is Ops.ADD and _is_offset(idx.src[1]): offset = imm(idx.src[1].dtype, idx.src[1].arg * scale)
  elif _is_offset(idx): offset = imm(idx.dtype, idx.arg * scale)
  else: offset = imm(dtypes.int32, 0)

  if idx.op is Ops.ADD and offset.arg != 0 and _is_soffset(idx.src[0]): soffset = idx.src[0] * scale
  elif _is_soffset(idx): soffset = idx * scale
  else: soffset = UOp(Ops.NOOP)
  
  return (base, soffset, offset)

# [vgpr + u16]
def lds_address(x:UOp) -> tuple[UOp, UOp]: pass

# [vgpr pair(u64) + const(i13)] or [sgpr pair(u64) + vgpr(u32) + const(i13)] or [sgpr pair(u64) + const(i13) + threadid * 4]
def global_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  base, idx = x.src
  scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  assert is_uniform(base) and not is_uniform(idx)

  def is_saddr(x:UOp): return not x.overflows(dtypes.uint32)
  def _is_offset(x:UOp): return x.op is Ops.CONST and -(1 << 12) <= x.arg * scale < (1 << 12)

  # TODO: idx.src[0] here must be in a vgpr, need to insert a move if it isn't
  if idx.op is Ops.ADD and is_saddr(x.src[0] * scale) and _is_offset(idx.src[1]): return (base, idx.src[0] * scale, imm(x.dtype, x.arg * scale))
  if is_saddr(idx * scale): return (base, idx * scale, imm(dtypes.int32, 0))
  # TODO: need to cast to 64bit (hi = lo >> 31) and add with carry, also check for the const
  lo = idx * scale
  hi = lo >> 31
  ret = UOp(Ops.TUPLE, x.dtype, (lo, hi))
  return (ret, UOp(Ops.NOOP), )

def lower_load(x:UOp):
  if is_uniform(x):
    return x.ins({32: RDNA3Ops.S_LOAD_B32, 64: RDNA3Ops.S_LOAD_B64, 128: RDNA3Ops.S_LOAD_B128,
                  256: RDNA3Ops.S_LOAD_B256, 512: RDNA3Ops.S_LOAD_B512}[x.dtype.bitsize], src=smem_address(x.src[0]))
  assert isinstance(x.src[0].dtype, PtrDType)
  if x.src[0].dtype.addrspace == AddrSpace.LOCAL:
    # TODO: ds_load_2addr_b32/64
    return x.ins({32: RDNA3Ops.DS_LOAD_B32, 64: RDNA3Ops.DS_LOAD_B64, 96: RDNA3Ops.DS_LOAD_B96,
                  128: RDNA3Ops.DS_LOAD_B128}[x.dtype.bitsize], src=lds_address(x.src[0]))
  if x.src[0].dtype.addrspace == AddrSpace.GLOBAL:
    return x.ins({32: RDNA3Ops.GLOBAL_LOAD_B32, 64: RDNA3Ops.GLOBAL_LOAD_B64, 96: RDNA3Ops.GLOBAL_LOAD_B96,
                  128: RDNA3Ops.S_LOAD_B128}[x.dtype.bitsize], src=global_address(x.src[0]))
  return x.ins({4: RDNA3Ops.GLOBAL_LOAD_B32}[x.dtype.itemsize])

def lower_store(x:UOp):
  return x.ins({4: RDNA3Ops.GLOBAL_STORE_B32}[x.dtype.itemsize])

isel += [(UPat(Ops.LOAD, src=(UPat(),), name="x"), lower_load),
         (UPat(Ops.STORE, src=(UPat(),), name="x"), lower_store)]

# TODO: not quite right and some missing cases?
def load_const(ctx:IselContext, x:UOp) -> UOp:
  if x.dtype.itemsize <= 32:
    if dtypes.int16.min <= x.arg <= dtypes.int16.max: return x.ins(RDNA3Ops.S_MOVK_I32, src=(imm(x.dtype, x.arg),), tag=ctx.vreg(SGPR))
    if dtypes.int32.min <= x.arg <= dtypes.int32.max: return x.ins(RDNA3Ops.S_MOV_B32, src=(imm(x.dtype, x.arg),), tag=ctx.vreg(SGPR))
  # otherwise need to emit two instructions
  # TODO: could emit shorter encodings if each half fits in 16bits, call this function again in that case
  bits = struct.unpack('Q', struct.pack(unwrap(x.dtype.fmt), x.arg))[0]
  return UOp(Ops.TUPLE, x.dtype, (x.ins(RDNA3Ops.S_MOV_B32, dtype=dtypes.uint32, src=(imm(dtypes.uint32, lo32(bits)))),
                                  x.ins(RDNA3Ops.S_MOV_B32, dtype=dtypes.uint32, src=(imm(dtypes.uint32, hi32(bits))))), tag=ctx.vreg(S64))

# extra
isel += [
  (UPat(Ops.SPECIAL, name="x"), lambda x: x.replace(src=(x.src[0].rtag(),)) if not x.src[0].tag else None),
  # STACK becomes TUPLE which enforces an alignment and ordering constraint on the src registers
  # TUPLE defines a virtual and tag its srcs as not defining virtuals, instead they define a sub virtual which is not allocatable
  # in regalloc
  (UPat(Ops.STACK, name="x"), lambda ctx,x: x.replace(op=Ops.TUPLE, tag=ctx.vreg(V128))),
  # GEP becomes GETTUPLE which retrieves the register at position x.arg from the multi definition src
  (UPat(Ops.LOAD).gep(name="x"), lambda x: x.replace(op=Ops.GETTUPLE)),
  # constants that must be moved to registers
  (UPat.cvar("x"), lambda ctx,x: load_const(ctx, x) if not x.tag else None),
]

isel_matcher = PatternMatcher(isel)

# ***** RDNA3 instruction encoding *****

def enc_ssrc(x:UOp) -> int:
  if x.op is Ops.CONST:
    v = x.arg
    if x.dtype in dtypes.ints+(dtypes.bool,):
      if 0 <= v <= 64: return 128 + v
      if -16 <= v <= -1: return 192 + (-v)
    else:
      if v == 0.0: return 128
      if v == 0.5: return 240
      if v == -0.5: return 241
      if v == 1.0: return 242
      if v == -1.0: return 243
      if v == 2.0: return 244
      if v == -2.0: return 245
      if v == 4.0: return 246
      if v == -4.0: return 247
      # TODO: prob not right cause of truncation
      if v == 1/(2*math.pi): return 248
    return 255

  assert isinstance(v:=x.reg, Register)
  if v is VCC: return 106
  if v is EXEC: return 126
  if v is SCC: return 253
  if v in SGPR: return v.index

def enc_sopk(x:UOp, opc:int) -> bytes: pass

def enc_sop1(x:UOp, opc:int) -> bytes:
  pass

def enc_sop2(x:UOp, opc:int) -> bytes:
  ssrc0 = enc_ssrc(x.src[0])
  ssrc1 = enc_ssrc(x.src[1])
  sdst = enc_ssrc(x)
  enc = struct.pack('<I', opc << 23 | sdst << 16 | ssrc1 << 8 | ssrc0)
  if ssrc0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif ssrc1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  return enc

def enc_sopc(x:UOp, opc:int) -> bytes:
  ssrc0 = enc_ssrc(x.src[0])
  ssrc1 = enc_ssrc(x.src[1])
  enc = struct.pack('<I', 0b101111110 << 23 | opc << 16 | ssrc1 << 8 | ssrc0)
  if ssrc0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif ssrc1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  return enc

def enc_src(x:UOp) -> int:
  if isinstance(v:=x.reg, Register) and v in VGPR: return 256 + v.index
  return enc_ssrc(x)

def enc_vop3(x:UOp, opc:int) -> bytes:
  src0 = enc_src(x.src[0])
  src1 = enc_src(x.src[1]) if len(x.src) > 1 else 124
  src2 = enc_src(x.src[2]) if len(x.src) > 2 else 124
  vdst = cast(Register, x.reg).index
  neg = sum(s.arg is Ops.NEG << i for i,s in enumerate(x.src))
  enc = struct.pack('<I', neg << 61 | 0 << 59 | src2 << 50 | src1 << 41 | src0 << 32 | 0b110101 << 26 | opc << 16 | 0 << 15 | 0 << 11 | 0 << 8 | vdst)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif src1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  elif src2 == 255: enc += struct.pack(unwrap(x.src[2].dtype.fmt), x.src[2].arg)
  return enc

def enc_vop2(x:UOp, opc:int) -> bytes:
  # TODO: if vop3 encode as vop3
  if any(s.arg is Ops.NEG for s in x.src): return enc_vop3(x, opc)
  src0 = enc_src(x.src[0])
  vsrc1 = cast(Register, x.src[1].reg).index
  vdst = cast(Register, x.reg).index
  enc = struct.pack('<I', opc << 25 | vdst << 17 | vsrc1 << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  return enc

def enc_vopc(x:UOp, opc:int) -> bytes:
  src0 = enc_src(x.src[0])
  vsrc1 = cast(Register, x.src[1].reg).index
  enc = struct.pack('<I', 0b111110 << 25 | opc << 17 | vsrc1 << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  return enc

def enc_vop1(x:UOp, opc:int) -> bytes:
  src0 = enc_src(x.src[0])
  vdst = cast(Register, x.reg).index
  enc = struct.pack('<I', 0b111111 << 25 | vdst << 17 | opc << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  return enc

encodings = {
  # SOPK
  RDNA3Ops.S_MOVK_I32: partial(enc_sopk, 2),
  # SOP1
  RDNA3Ops.S_SEXT_I32_I8: partial(enc_sop1, 14), RDNA3Ops.S_SEXT_I32_I16: partial(enc_sop1, 15),
  RDNA3Ops.S_MOV_B32: partial(enc_sop1, 0),
  # SOPC
  RDNA3Ops.S_CMP_EQ_I32: lambda x: enc_sopc(x, 0), RDNA3Ops.S_CMP_EQ_U32: lambda x: enc_sopc(x, 6), RDNA3Ops.S_CMP_EQ_U64: lambda x: enc_sopc(x, 16),
  RDNA3Ops.S_CMP_LG_I32: lambda x: enc_sopc(x, 1), RDNA3Ops.S_CMP_LG_U32: lambda x: enc_sopc(x, 7), RDNA3Ops.S_CMP_LG_U64: lambda x: enc_sopc(x, 17),
  RDNA3Ops.S_CMP_LT_I32: lambda x: enc_sopc(x, 4), RDNA3Ops.S_CMP_LT_U32: lambda x: enc_sopc(x, 10),
  # SOP2
  RDNA3Ops.S_ADD_U32: partial(enc_sop2, 0), RDNA3Ops.S_SUB_U32: partial(enc_sop2, 1), RDNA3Ops.S_MUL_I32: partial(enc_sop2, 44),
  RDNA3Ops.S_AND_B32: partial(enc_sop2, 22), RDNA3Ops.S_AND_B64: partial(enc_sop2, 23), RDNA3Ops.S_OR_B32: partial(enc_sop2, 24),
  RDNA3Ops.S_OR_B64: partial(enc_sop2, 25), RDNA3Ops.S_XOR_B32: partial(enc_sop2, 26), RDNA3Ops.S_XOR_B64: partial(enc_sop2, 27),
  RDNA3Ops.S_LSHL_B32: partial(enc_sop2, 8), RDNA3Ops.S_LSHL_B64: partial(enc_sop2, 9), RDNA3Ops.S_LSHR_B32: partial(enc_sop2, 10),
  RDNA3Ops.S_LSHR_B64: partial(enc_sop2, 11), RDNA3Ops.S_ASHR_I32: partial(enc_sop2, 12), RDNA3Ops.S_ASHR_I64: partial(enc_sop2, 13),
  RDNA3Ops.S_MIN_I32: partial(enc_sop2, 18), RDNA3Ops.S_MIN_U32: partial(enc_sop2, 19),
  RDNA3Ops.S_MAX_I32: partial(enc_sop2, 20), RDNA3Ops.S_MAX_U32: partial(enc_sop2, 21),
  RDNA3Ops.S_CSELECT_B32: partial(enc_sop2, 48), RDNA3Ops.S_CSELECT_B64: partial(enc_sop2, 49),
  # VOP1
  RDNA3Ops.V_EXP_F16: lambda x: enc_vop1(x, 88), RDNA3Ops.V_EXP_F32: lambda x: enc_vop1(x, 37), RDNA3Ops.V_LOG_F16: lambda x: enc_vop1(x, 87),
  RDNA3Ops.V_LOG_F32: lambda x: enc_vop1(x, 39), RDNA3Ops.V_SIN_F16: lambda x: enc_vop1(x, 96), RDNA3Ops.V_SIN_F32: lambda x: enc_vop1(x, 53),
  RDNA3Ops.V_RCP_F16: lambda x: enc_vop1(x, 84), RDNA3Ops.V_RCP_F32: lambda x: enc_vop1(x, 42), RDNA3Ops.V_RCP_F64: lambda x: enc_vop1(x, 47),
  RDNA3Ops.V_SQRT_F16: lambda x: enc_vop1(x, 85), RDNA3Ops.V_SQRT_F32: lambda x: enc_vop1(x, 51), RDNA3Ops.V_SQRT_F64: lambda x: enc_vop1(x, 52),
  RDNA3Ops.V_TRUNC_F16: lambda x: enc_vop1(x, 93), RDNA3Ops.V_TRUNC_F32: lambda x: enc_vop1(x, 33), RDNA3Ops.V_TRUNC_F64: lambda x: enc_vop1(x, 23),
  RDNA3Ops.V_CVT_F16_U16: lambda x: enc_vop1(x, 80), RDNA3Ops.V_CVT_F16_I16: lambda x: enc_vop1(x, 81), RDNA3Ops.V_CVT_U16_F16: lambda x: enc_vop1(x, 82),
  RDNA3Ops.V_CVT_I16_F16: lambda x: enc_vop1(x, 83),
  RDNA3Ops.V_CVT_F32_U32: lambda x: enc_vop1(x, 6), RDNA3Ops.V_CVT_F32_I32: lambda x: enc_vop1(x, 5), RDNA3Ops.V_CVT_U32_F32: lambda x: enc_vop1(x, 7),
  RDNA3Ops.V_CVT_I32_F32: lambda x: enc_vop1(x, 8),
  RDNA3Ops.V_CVT_I32_F64: lambda x: enc_vop1(x, 3), RDNA3Ops.V_CVT_F64_I32: lambda x: enc_vop1(x, 4), RDNA3Ops.V_CVT_U32_F64: lambda x: enc_vop1(x, 21),
  RDNA3Ops.V_CVT_F64_U32: lambda x: enc_vop1(x, 22),
  RDNA3Ops.V_CVT_F16_F32: lambda x: enc_vop1(x, 10), RDNA3Ops.V_CVT_F32_F16: lambda x: enc_vop1(x, 11), RDNA3Ops.V_CVT_F32_F64: lambda x: enc_vop1(x, 15),
  RDNA3Ops.V_CVT_F64_F32: lambda x: enc_vop1(x, 16),
  RDNA3Ops.V_CVT_I32_I16: lambda x: enc_vop1(x, 106), RDNA3Ops.V_CVT_U32_U16: lambda x: enc_vop1(x, 107),
  # VOP2
  RDNA3Ops.V_ADD_NC_U32: lambda x: enc_vop2(x, 37), RDNA3Ops.V_SUB_NC_U32: lambda x: enc_vop2(x, 38), RDNA3Ops.V_SUBREV_NC_U32: lambda x: enc_vop2(x, 39),
  RDNA3Ops.V_LSHLREV_B32: lambda x: enc_vop2(x, 24), RDNA3Ops.V_LSHRREV_B32: lambda x: enc_vop2(x, 25), RDNA3Ops.V_ASHRREV_I32: lambda x: enc_vop2(x, 26),
  RDNA3Ops.V_ADD_F16: lambda x: enc_vop2(x, 50), RDNA3Ops.V_ADD_F32: lambda x: enc_vop2(x, 3),
  RDNA3Ops.V_MUL_F16: lambda x: enc_vop2(x, 53), RDNA3Ops.V_MUL_F32: lambda x: enc_vop2(x, 8),
  RDNA3Ops.V_SUB_F16: lambda x: enc_vop2(x, 51), RDNA3Ops.V_SUB_F32: lambda x: enc_vop2(x, 4),
  RDNA3Ops.V_SUBREV_F16: lambda x: enc_vop2(x, 52), RDNA3Ops.V_SUBREV_F32: lambda x: enc_vop2(x, 5),
  RDNA3Ops.V_MAX_F16: lambda x: enc_vop2(x, 57), RDNA3Ops.V_MAX_F32: lambda x: enc_vop2(x, 16),
  RDNA3Ops.V_MAX_I32: lambda x: enc_vop2(x, 18), RDNA3Ops.V_MAX_U32: lambda x: enc_vop2(x, 20),
  RDNA3Ops.V_MIN_F16: lambda x: enc_vop2(x, 58), RDNA3Ops.V_MIN_F32: lambda x: enc_vop2(x, 15),
  RDNA3Ops.V_MIN_I32: lambda x: enc_vop2(x, 17), RDNA3Ops.V_MIN_U32: lambda x: enc_vop2(x, 19),
  RDNA3Ops.V_AND_B32: lambda x: enc_vop2(x, 27), RDNA3Ops.V_OR_B32: lambda x: enc_vop2(x, 28), RDNA3Ops.V_XOR_B32: lambda x: enc_vop2(x, 29),
  # TODO: these don't have vop3 encoding and that should be enforced
  RDNA3Ops.V_FMAC_F16: lambda x: enc_vop2(x, 54), RDNA3Ops.V_FMAC_F32: lambda x: enc_vop2(x, 43),
  RDNA3Ops.V_FMAMK_F16: lambda x: enc_vop2(x, 55), RDNA3Ops.V_FMAMK_F32: lambda x: enc_vop2(x, 44),
  RDNA3Ops.V_FMAAK_F16: lambda x: enc_vop2(x, 56), RDNA3Ops.V_FMAAK_F32: lambda x: enc_vop2(x, 45),
  # VOPC
  RDNA3Ops.V_CMP_LT_F16: lambda x: enc_vopc(x, 1), RDNA3Ops.V_CMP_LT_F32: lambda x: enc_vopc(x, 17), RDNA3Ops.V_CMP_LT_F64: lambda x: enc_vopc(x, 33),
  RDNA3Ops.V_CMP_EQ_F16: lambda x: enc_vopc(x, 2), RDNA3Ops.V_CMP_EQ_F32: lambda x: enc_vopc(x, 18), RDNA3Ops.V_CMP_EQ_F64: lambda x: enc_vopc(x, 34),
  RDNA3Ops.V_CMP_NEQ_F16: lambda x: enc_vopc(x, 13), RDNA3Ops.V_CMP_NEQ_F32: lambda x: enc_vopc(x, 29), RDNA3Ops.V_CMP_NEQ_F64: lambda x: enc_vopc(x, 45),
  RDNA3Ops.V_CMP_LT_I16: lambda x: enc_vopc(x, 49), RDNA3Ops.V_CMP_LT_I32: lambda x: enc_vopc(x, 65), RDNA3Ops.V_CMP_LT_I64: lambda x: enc_vopc(x, 81),
  RDNA3Ops.V_CMP_LT_U16: lambda x: enc_vopc(x, 57), RDNA3Ops.V_CMP_LT_U32: lambda x: enc_vopc(x, 73), RDNA3Ops.V_CMP_LT_U64: lambda x: enc_vopc(x, 89),
  RDNA3Ops.V_CMP_EQ_I16: lambda x: enc_vopc(x, 50), RDNA3Ops.V_CMP_EQ_I32: lambda x: enc_vopc(x, 66), RDNA3Ops.V_CMP_EQ_I64: lambda x: enc_vopc(x, 82),
  RDNA3Ops.V_CMP_EQ_U16: lambda x: enc_vopc(x, 58), RDNA3Ops.V_CMP_EQ_U32: lambda x: enc_vopc(x, 74), RDNA3Ops.V_CMP_EQ_U64: lambda x: enc_vopc(x, 90),
  RDNA3Ops.V_CMP_NE_I16: lambda x: enc_vopc(x, 53), RDNA3Ops.V_CMP_NE_I32: lambda x: enc_vopc(x, 69), RDNA3Ops.V_CMP_NE_I64: lambda x: enc_vopc(x, 85),
  RDNA3Ops.V_CMP_NE_U16: lambda x: enc_vopc(x, 61), RDNA3Ops.V_CMP_NE_U32: lambda x: enc_vopc(x, 77), RDNA3Ops.V_CMP_NE_U64: lambda x: enc_vopc(x, 93),
  # VOP3
  RDNA3Ops.V_FMA_F16: partial(enc_vop3, 584), RDNA3Ops.V_FMA_F32: partial(enc_vop3, 531), RDNA3Ops.V_FMA_F64: partial(enc_vop3, 532),
  RDNA3Ops.V_AND_B16: partial(enc_vop3, 866), RDNA3Ops.V_XOR_B16: partial(enc_vop3, 868), RDNA3Ops.V_OR_B16: partial(enc_vop3, 867),
  RDNA3Ops.V_LSHLREV_B16: partial(enc_vop3, 824), RDNA3Ops.V_LSHLREV_B64: partial(enc_vop3, 828),
  RDNA3Ops.V_LSHRREV_B16: partial(enc_vop3, 825), RDNA3Ops.V_LSHRREV_B64: partial(enc_vop3, 829),
  RDNA3Ops.V_ASHRREV_I16: partial(enc_vop3, 826), RDNA3Ops.V_ASHRREV_I64: partial(enc_vop3, 830),
  RDNA3Ops.V_MAX_I16: partial(enc_vop3, 778), RDNA3Ops.V_MAX_U16: partial(enc_vop3, 777), RDNA3Ops.V_MAX_F64: partial(enc_vop3, 810),
  RDNA3Ops.V_MIN_I16: partial(enc_vop3, 780), RDNA3Ops.V_MIN_U16: partial(enc_vop3, 779), RDNA3Ops.V_MIN_F64: partial(enc_vop3, 809),
}

# final rewrite to match the isa spec
post_regalloc_matcher = PatternMatcher([
  # these are two address, if regalloc didn't coalesce downgrade them to their long encoding version
  (UPat(Ops.INS, arg=RDNA3Ops.V_FMAC_F16, name="x"), lambda x: (nx:=x.replace(arg=RDNA3Ops.V_FMA_F16), [nx]) if x.reg != x.src[0].reg else None),
  (UPat(Ops.INS, arg=RDNA3Ops.V_FMAC_F32, name="x"), lambda x: (nx:=x.replace(arg=RDNA3Ops.V_FMA_F32), [nx]) if x.reg != x.src[0].reg else None),
])

class RDNA3Renderer(ISARenderer):
  has_local = True
  extra_matcher = extra_matcher
  pre_isel_matcher = PatternMatcher([])
  isel_matcher = isel_matcher
  post_regalloc_matcher = post_regalloc_matcher
  code_for_op = {x: lambda: None for x in (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.FDIV, Ops.CMPLT, Ops.CMPEQ)}

  def is_two_address(self, x:UOp) -> bool: return x.arg in RDNA3GroupOp.TwoAddress
  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+dtypes.int8s+dtypes.int64s}

  def render(self, uops:list[UOp]) -> str:
    targets: dict[str, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()
    max_sgpr, max_vgpr = 0, 0
    n_bufs, n_vars, lds_size, gids = 0, 0, 0, set()

    # encode the instructions and record kernel descriptors
    for u in uops:
      if u.op is Ops.PARAM: n_bufs += 1
      elif u.op is Ops.DEFINE_VAR: n_vars += 1
      elif u.op is Ops.DEFINE_LOCAL: lds_size += u.ptrdtype.size * u.ptrdtype.base.itemsize
      elif u.op is Ops.SPECIAL and u.arg.startswith("gidx"): gids.add(int(u.arg[-1]))
      if u.op is not Ops.INS: continue
      if isinstance(reg:=u.reg, Register):
        if reg in (V32, V64): max_vgpr = max(max_vgpr, reg.index + u.dtype.itemsize)
        elif reg in (S32, S64): max_sgpr = max(max_sgpr, reg.index + u.dtype.itemsize)
      #if u.arg is X86Ops.LABEL:
      #  targets[u.tag] = len(binary)
      #  continue
      #if u.arg not in encodings or (l:=encodings[u.arg](u)) is None:
      #  raise RuntimeError(f"failed to encode {u.arg} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      #binary.extend(l)
      #if u.arg in (X86Ops.JL, X86Ops.JB, X86Ops.JE, X86Ops.JNE, X86Ops.JGE, X86Ops.JMP): jumps[u] = len(binary)
    # fixup jump targets now that encoding size is known
    for u in uops:
      if (t:=jumps.get(u)) is not None: binary[t-4:t] = (targets[u.tag] - t).to_bytes(4, 'little', signed=True)

    # build the elf code object
    # ** pad text to ISA alignment
    padding_inst = 0 #s_code_end().to_bytes()
    text = binary + padding_inst * ((hsa.AMD_ISA_ALIGN_BYTES - len(binary) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES)
    text_offset = round_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
    # ** pack kernel descriptor (rodata)
    next_free_vgpr = round_up(max_vgpr, 8)
    vgpr_granule = max(0, (next_free_vgpr + 7) // 8 - 1)
    sgpr_granule = 0
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
    desc.group_segment_fixed_size = lds_size
    desc.kernarg_size = n_bufs * 8 + n_vars * 4
    desc.kernel_code_entry_byte_offset = -len(text)

    # https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc1-gfx6-gfx12-table
    # NOTE: CU mode is the default
    desc.compute_pgm_rsrc1 = (vgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT |
                              sgpr_granule << amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT |
                              3 << amdgpu_kd.COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64_SHIFT |
                              1 << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP_SHIFT |
                              1 << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE_SHIFT |
                              1 << amdgpu_kd.COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED_SHIFT)
    desc.compute_pgm_rsrc2 = (2 << amdgpu_kd.COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT |
                              int(0 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT |
                              int(1 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT |
                              int(2 in gids) << amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT)
    desc.kernel_code_properties = (1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT |
                                   1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT)
    rodata = bytes(desc)

    # ** pack ELF
    sh_names:list[int] = []
    strtab = bytearray(b"\x00")
    for name in [".text", ".rodata", ".strtab"]:
      sh_names.append(len(strtab))
      strtab += name.encode("ascii") + b"\x00"

    rodata_offset = round_up(text_offset + (text_size := len(text)), hsa.AMD_KERNEL_CODE_ALIGN_BYTES)
    strtab_offset = rodata_offset + (rodata_size := len(rodata))
    shdr_offset   = strtab_offset + (strtab_size := len(strtab))

    sections = [(libc.SHT_PROGBITS, libc.SHF_ALLOC | libc.SHF_EXECINSTR, text_offset, text_offset, text_size),
                (libc.SHT_PROGBITS, libc.SHF_ALLOC, rodata_offset, rodata_offset, rodata_size),
                (libc.SHT_STRTAB, 0, 0, strtab_offset, strtab_size)]
    shdrs = (libc.Elf64_Shdr * len(sections))()
    for i, s in enumerate(sections): shdrs[i] = libc.Elf64_Shdr(sh_names[i], *s)

    ehdr = libc.Elf64_Ehdr()
    ehdr.e_ident[:5], ehdr.e_shoff, ehdr.e_shnum, ehdr.e_shstrndx = b"\x7FELF\x02", shdr_offset, len(sections), 2

    elf = bytearray(shdr_offset + ctypes.sizeof(shdrs))
    elf[0:ctypes.sizeof(ehdr)] = bytes(ehdr)
    elf[text_offset:text_offset+text_size] = text
    elf[rodata_offset:rodata_offset+rodata_size] = rodata
    elf[strtab_offset:strtab_offset+strtab_size] = strtab
    elf[shdr_offset:shdr_offset+ctypes.sizeof(shdrs)] = bytes(shdrs)
    binary = bytes(elf)

    return binary.hex()
