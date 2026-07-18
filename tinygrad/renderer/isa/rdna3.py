# flake8: noqa: E702
# allow semicolons to put multiple ops on one line
import struct, math, ctypes, types
from functools import cache, partial
from typing import cast
from tinygrad.dtype import dtypes, PtrDType, DType, truncate, AddrSpace
from tinygrad.uop import FastEnum, auto, Ops, GroupOp
from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.renderer.isa import ISARenderer, IselContext, Register, FlagRematContext, PostRegallocContext, \
  imm, def_reg, reg_use, reg_uses, reg_defs
from tinygrad.helpers import unwrap, Target, lo32, hi32, round_up
from tinygrad.runtime.autogen import amdgpu_kd, hsa, libc

# https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0
# ***** RDNA3 Ops *****

class RDNA3Ops(FastEnum):
  LABEL = auto()
  # SOPK
  S_MOVK_I32 = auto()
  S_WAITCNT_VSCNT = auto()
  # SOP1
  S_SEXT_I32_I8 = auto(); S_SEXT_I32_I16 = auto()
  S_MOV_B32 = auto()
  # SOP2
  S_ADDC_U32 = auto()
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
  # SOPP
  S_NOP = auto()
  S_WAITCNT = auto()
  S_BRANCH = auto()
  S_CBRANCH_SCC0 = auto()
  S_ENDPGM = auto()
  # VOP1
  V_MOV_B32 = auto()
  V_READFIRSTLANE_B32 = auto()
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
  V_MUL_I32_I24 = auto(); V_MUL_U32_U24 = auto()
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
  V_CMP_GT_F32 = auto()
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
  V_MUL_LO_U16 = auto(); V_MUL_LO_U32 = auto()
  V_MUL_HI_U32 = auto()
  V_LSHLREV_B16 = auto(); V_LSHLREV_B64 = auto()
  V_LSHRREV_B16 = auto(); V_LSHRREV_B64 = auto()
  V_ASHRREV_I16 = auto(); V_ASHRREV_I64 = auto()
  V_MAX_I16 = auto(); V_MAX_U16 = auto(); V_MAX_F64 = auto()
  V_MIN_I16 = auto(); V_MIN_U16 = auto(); V_MIN_F64 = auto()
  V_BFE_U32 = auto()
  # SMEM
  S_LOAD_B32 = auto(); S_LOAD_B64 = auto(); S_LOAD_B128 = auto(); S_LOAD_B256 = auto(); S_LOAD_B512 = auto()
  # LDS
  DS_LOAD_U8 = auto(); DS_LOAD_I8 = auto(); DS_LOAD_U16 = auto(); DS_LOAD_I16 = auto()
  DS_LOAD_B32 = auto(); DS_LOAD_B64 = auto(); DS_LOAD_B96 = auto(); DS_LOAD_B128 = auto()
  DS_STORE_B8 = auto(); DS_STORE_B16 = auto(); DS_STORE_B32 = auto(); DS_STORE_B64 = auto(); DS_STORE_B96 = auto(); DS_STORE_B128 = auto()
  # SCRATCH
  SCRATCH_LOAD_U8 = auto(); SCRATCH_LOAD_I8 = auto(); SCRATCH_LOAD_U16 = auto(); SCRATCH_LOAD_I16 = auto()
  SCRATCH_LOAD_B32 = auto(); SCRATCH_LOAD_B64 = auto(); SCRATCH_LOAD_B96 = auto(); SCRATCH_LOAD_B128 = auto()
  SCRATCH_STORE_B8 = auto(); SCRATCH_STORE_B16 = auto(); SCRATCH_STORE_B32 = auto(); SCRATCH_STORE_B64 = auto()
  SCRATCH_STORE_B96 = auto(); SCRATCH_STORE_B128 = auto()
  # GLOBAL
  GLOBAL_LOAD_U8 = auto(); GLOBAL_LOAD_I8 = auto(); GLOBAL_LOAD_U16 = auto(); GLOBAL_LOAD_I16 = auto()
  GLOBAL_LOAD_B32 = auto(); GLOBAL_LOAD_B64 = auto(); GLOBAL_LOAD_B96 = auto(); GLOBAL_LOAD_B128 = auto()
  GLOBAL_STORE_B8 = auto(); GLOBAL_STORE_B16 = auto(); GLOBAL_STORE_B32 = auto(); GLOBAL_STORE_B64 = auto()
  GLOBAL_STORE_B96 = auto(); GLOBAL_STORE_B128 = auto()

class RDNA3GroupOp:
  # RDNA3Ops whose first src is also the destination
  TwoAddress = {RDNA3Ops.V_FMAC_F16, RDNA3Ops.V_FMAC_F32}
  # VOP1/2 RDNA3Ops that don't have a VOP3 version
  NO_VOP3 = {RDNA3Ops.V_FMAAK_F16, RDNA3Ops.V_FMAAK_F32, RDNA3Ops.V_FMAMK_F16, RDNA3Ops.V_FMAMK_F32}
  # RDNA3Ops that write to SCC
  WriteSCC = {RDNA3Ops.S_AND_B32, RDNA3Ops.S_XOR_B32, RDNA3Ops.S_OR_B32, RDNA3Ops.S_LSHL_B32, RDNA3Ops.S_LSHR_B32, RDNA3Ops.S_ASHR_I32,
              RDNA3Ops.S_AND_B64, RDNA3Ops.S_XOR_B64, RDNA3Ops.S_OR_B64, RDNA3Ops.S_LSHL_B64, RDNA3Ops.S_LSHR_B64, RDNA3Ops.S_ASHR_I64,
              RDNA3Ops.S_ADD_U32, RDNA3Ops.S_SUB_U32, RDNA3Ops.S_MAX_I32, RDNA3Ops.S_MAX_U32, RDNA3Ops.S_MIN_I32, RDNA3Ops.S_MIN_U32,
              RDNA3Ops.S_ADDC_U32}

# ***** RDNA3 legalization *****

def expand_cdivmod32(z:UOp, x:UOp, y:UOp) -> UOp:
  def mulhu(a:UOp, b:UOp) -> UOp: return (a.cast(dtypes.uint64) * b.cast(dtypes.uint64) >> 32).cast(dtypes.uint32)
  is_div, is_signed = z.op is Ops.CDIV, z.dtype in dtypes.sints
  if is_signed:
    sign_x, sign_y = x >> 31, y >> 31
    sign = sign_x ^ sign_y if is_div else sign_x
    x, y = (x + sign_x) ^ sign_x, (y + sign_y) ^ sign_y
  # everything below is unsigned
  x, y = x.cast(dtypes.uint32), y.cast(dtypes.uint32)
  # initial reciprocal estimate
  scale = UOp.const(dtypes.float32, struct.unpack('<f', struct.pack('<I', 0x4F7FFFFE))[0])
  z = (y.cast(dtypes.float32).reciprocal() * scale).cast(dtypes.uint32)
  # one round of UNR
  z = z + mulhu(z, (0 - y) * z)
  # quotient/remainder estimate
  q = mulhu(x, z)
  r = x - q * y
  # first refinement (always updates r, updates q only for div)
  cond = r >= y
  if is_div: q = cond.where(q + 1, q)
  r = cond.where(r - y, r)
  # second refinement (result comes from q for div, r for rem)
  cond = r >= y
  res = cond.where(q + 1, q) if is_div else cond.where(r - y, r)
  # re-apply sign
  if is_signed: res = (res.cast(dtypes.int32) ^ sign) - sign
  return res

extra_matcher = PatternMatcher([
  (UPat((Ops.CDIV, Ops.CMOD), dtypes.int32s, (UPat.var("x"), UPat.var("y")), name="z"), expand_cdivmod32),
  # bool CMPNE is XOR, bool CMPEQ is XOR+XOR, bool CMPLT is XOR+AND
  (UPat.var('x', dtypes.bool).ne(UPat.var('y')), lambda x,y: x^y),
  (UPat.var('x', dtypes.bool).alu(Ops.CMPEQ, UPat.var('y')), lambda x,y: (x^y)^True),
  (UPat.var('x', dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
  # rewrite -x -> 0 - x for ints, for floats isel tries to fold the neg
  (UPat.var("x", dtypes.ints).alu(Ops.NEG), lambda x: x.const_like(0).alu(Ops.SUB, x)),
  # cast to pointer is a noop
  (UPat.var("y").cast(name="x"), lambda y,x: y if isinstance(x.dtype, PtrDType) or y.dtype == dtypes.void else None),
  # TODO: this isn't actually required, just a noop
  #(UPat.var("y", dtypes.bool).cast().named("x"), lambda y,x: y.where(x.const_like(1), x.const_like(0))),
  # no support for 8bit ops, promote them to 32bit, sign sensitive ops require masking
  # TODO: int8s and bool need to be treated differently
  # TODO: load int8s need a cast to int32
  (UPat(GroupOp.ALU, dtypes.int8s, name="x"), lambda x: x.replace(dtype=dtypes.int32)),
])

# ***** RDNA3 pre instruction selection *****

pre_isel_matcher = PatternMatcher([
  # DEFINE_REG becomes DEFINE_PRIVATE, but really it should be a PHI node
  (UPat(Ops.DEFINE_REG, name="x"), lambda x:
   x.replace(op=Ops.DEFINE_PRIVATE, dtype=x.dtype.base.ptr(x.dtype.size, AddrSpace.PRIVATE)) if isinstance(x.arg, int) else None),
  (UPat(Ops.INDEX, name="x"), lambda x:
   x.replace(dtype=x.dtype.base.ptr(x.dtype.size, AddrSpace.PRIVATE)) if x.ptrdtype.addrspace == AddrSpace.REG else None),
])

# ***** RDNA3 registers *****

SCC = Register("scc", 253, 1)
VCC_LO = Register("vcc_lo", 106, 32)
VCC_HI = Register("vcc_hi", 107, 32)
EXEC_LO = Register("exec_lo", 126, 32)
EXEC_HI = Register("exec_hi", 127, 32)

# scalar registers, the range step is the required alignment
S32 = tuple(Register(f"s{i}", i, 32) for i in range(106))
S64 = tuple(Register(f"s[{i}:{i+2-1}]", i, 64, S32[i:i+2]) for i in range(0, 106, 2))
S128 = tuple(Register(f"s[{i}:{i+4-1}]", i, 128, S32[i:i+4]) for i in range(0, 106, 4))
S256 = tuple(Register(f"s[{i}:{i+8-1}]", i, 256, S32[i:i+8]) for i in range(0, 106, 4))
S512 = tuple(Register(f"s[{i}:{i+32-1}]", i, 512, S32[i:i+32]) for i in range(0, 106, 4))
# 8 here is only for dtypes.bool
sz_to_sgpr = {8: S32, 32: S32, 64: S64, 128: S128, 256: S256, 512: S512}

# vector registers, the smallest unit is 16bit as instructions may specify access to the low or high 16bits of a vgpr
V16 = tuple(Register(f"v{i//2}.{'lh'[i%2]}", i//2+256*(i%2), 16) for i in range(512))
V32 = tuple(Register(f"v{i//2}", i//2, 32, V16[i:i+2]) for i in range(0, 512, 2))
V64 = tuple(Register(f"v[{i//2}:{i//2+2-1}]", i//2, 64, V16[i:i+4]) for i in range(0, 512, 2))
V128 = tuple(Register(f"v[{i//2}:{i//2+4-1}]", i//2, 128, V16[i:i+8]) for i in range(0, 512, 2))
sz_to_vgpr = {16: V16, 32: V32, 64: V64, 128: V128}

# ***** RDNA3 instruction selection ****
# rdna3 contains 2 separate register classes, sgpr and vgpr. sgpr are used and defined by sop instructions,
# these require the operation to be uniform, meaning the result is the same across lanes in a wave.
# when the op isn't uniform or when the op,dtype pair has no sop lowering vop instructions are used.
# these use and define vgprs but they can also use sgprs with some restrictions.
# this means we need to manage transitions between the scalar and vector domains, additionally we have to manage
# the different way bools are modeled in the scalar vs vector domain, we do this by using tags.
# consumers can tag srcs specifying where that consumer requires that src to be. The list of tags is the following:
# "vgpr" means consumer requires x to be in a vgpr, if x is sop a move to vgpr is inserted
# "sgpr" means consumer requires x to be in a sgpr, if x is vop a move to sgpr is inserted
# "scc" means consumer requires x to be a scalar bool, if x is vop a conversion from vector to scalar bool is inserted
# "vcc" means consumer requires x to be a vector bool, if x is sop a conversion from scalar to vector bool is inserted
# the tags need to be wiped when lowering to RDNA3Op otherwise duplicate instructions can be introduced

# TODO: this should be introduced in regalloc as a spill when exec is clobbered
exec_use = def_reg(dtypes.uint32, EXEC_LO)
#exec_use = UOp(Ops.INS, dtypes.uint32, (def_reg(dtypes.uint32, EXEC_LO),), RDNA3Ops.S_AND_SAVEEXEC_B32, (S32[4], EXEC_LO, SCC))

def fits_in_bits(x:UOp, bits:int, signed:bool|None=None) -> bool:
  if signed is False or signed is None and dtypes.is_unsigned(x.dtype): return 0 <= x.vmin and x.vmax < (1 << bits)
  return -(1 << (bits - 1)) <= x.vmin and x.vmax < (1 << (bits - 1))

def is_inline_const(c:UOp) -> bool:
  if c.op is not Ops.CONST: return False
  if c.dtype in dtypes.ints: return -16 <= c.arg <= 64
  if c.dtype in dtypes.floats: return any(c.arg == v for v in (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2*math.pi)))

# a literal must fit in 32bits
def to_literal(c:UOp) -> UOp|None:
  if c.op is not Ops.CONST: return None
  if c.dtype is dtypes.int64: return imm(dtypes.int32, c.arg) if not c.overflows(dtypes.int32) else None
  if c.dtype is dtypes.uint64: return imm(dtypes.uint32, c.arg) if not c.overflows(dtypes.uint32) else None
  if c.dtype is dtypes.float64: return imm(dtypes.float64, c.arg) if lo32(struct.unpack('Q', struct.pack('d', c.arg))[0]) == 0 else None
  return imm(c.dtype, c.arg)

def vreg_sgpr(ctx:IselContext, x:UOp) -> tuple[Register]:
  sz = (x.dtype.bitsize if x.dtype.bitsize != 1 else 32) if not isinstance(x.dtype, PtrDType) else 64
  return (ctx.vreg(sz_to_sgpr[sz]),)

def vreg_vgpr(ctx:IselContext, x:UOp) -> tuple[Register]:
  width = (x.dtype.bitsize if x.dtype.bitsize != 1 else 32) if not isinstance(x.dtype, PtrDType) else 64
  return (ctx.vreg(sz_to_vgpr[width]),)
# TODO: support 64bit with TUPLE
def sgpr_to_vgpr(ctx:IselContext, x:UOp) -> UOp: return x.ins(RDNA3Ops.V_MOV_B32, src=(x.rtag(None),))

@cache
def is_uniform(x:UOp) -> bool:
  if x.op is Ops.SPECIAL and x.arg.startswith("lidx"): return False
  return all(is_uniform(s) for s in x.src)

def handle_user(x:UOp) -> UOp|None:
  if x.tag == "vgpr": return None
  # if user requires 
  if x.tag == "scc": return None
  if x.tag == "vcc": return None
  return None

# this is used when any src can be scalar (sgpr/inline const/literal), which is the case for SOP and VOP3 instructions
def ssrcs(src:tuple[UOp, ...]) -> tuple[UOp, ...]:
  # any src can be an inline constant, only one can be a literal
  ret, free_literal = [], True
  for s in src:
    if (sl:=to_literal(s)) is not None and (free_literal or is_inline_const(sl)):
      if not is_inline_const(sl): free_literal = False
      s = sl
    ret.append(s)
  return tuple(ret)

def sop_src(src:tuple[UOp, ...]) -> tuple[UOp, ...]: return tuple(s.rtag("sgpr") if s.op in GroupOp.Elementwise|{Ops.LOAD} else s for s in ssrcs(src))
def lower_sop1(op:RDNA3Ops, x:UOp) -> UOp: return x.ins(op, src=sop_src(x.src))
def lower_sop2(op:RDNA3Ops, x:UOp, a:UOp, b:UOp) -> UOp: return x.ins(op, src=sop_src((a, b)))
def lower_s_cselect(op:RDNA3Ops, x:UOp, a:UOp, b:UOp, c:UOp) -> UOp: return x.ins(op, src=sop_src((a, b, c.rtag("scc"))))

# NOTE: any SOP instruction can read SCC directly, S_CSELECT_B32/64 MUST read SCC
# do this in a separate pass that adds the extra required moves
def lower_sopc(op:RDNA3Ops, x:UOp) -> UOp:
  # if the user doesn't require SCC specifically insert a move to free up SCC
  if x.tag == "sgpr":
    ret = x.rtag(None)
    return ret.ins(RDNA3Ops.S_MOV_B32, src=(ret,))
  return x.ins(op, src=sop_src(x.src))

# all sopc 64bit comparisons are sign agnostic unless a negative literal is present as
# literals in 64bit instructions are extended according to the sign of the instruction and sign ext != zero ext for negative ints
def lower_sopc_64bit(op:RDNA3Ops, x:UOp) -> UOp|None:
  # an int >= -16 and <= 64 becomes an inline constant instead of a literal
  if x.dtype is dtypes.int64 and x.src[1].op is Ops.CONST and dtypes.int32.min <= x.src[1].arg < -16: return None
  return lower_sopc(x, op)

# [base(sgpr pair) + soffset(sgpr) + offset]
def lower_smem(x:UOp) -> UOp|None:
  sz = x.dtype.itemsize if not isinstance(x.dtype, PtrDType) else 8
  if x.src[0].ptrdtype.addrspace != AddrSpace.GLOBAL or sz < 4: return None
  op = {32: RDNA3Ops.S_LOAD_B32, 64: RDNA3Ops.S_LOAD_B64, 128: RDNA3Ops.S_LOAD_B128,
        256: RDNA3Ops.S_LOAD_B256, 512: RDNA3Ops.S_LOAD_B512}[sz*8]
  base, idx = x.src[0].src
  scale = 8 if isinstance(x.dtype, PtrDType) else base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1

  #if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST:
  #  soffset = idx.src[0] * scale
  #  offset = imm(idx.src[1].dtype, idx.src[1].arg * scale)
  #  if fits_in_bits(soffset, 32, False) and fits_in_bits(offset, 21, True): return x.ins(op, src=(base, soffset, offset))

  if idx.op is Ops.CONST:
    offset = imm(idx.dtype, idx.arg * scale)
    if fits_in_bits(offset, 21, True): return x.ins(op, src=(base, UOp(Ops.NOOP), offset))

  soffset = idx * scale
  if fits_in_bits(soffset, 32, False): return x.ins(op, src=(base, soffset, imm(dtypes.int32, 0)))

  return x.ins(op, src=(x.src[0], UOp(Ops.NOOP), imm(dtypes.int32, 0)))

def lower_index(ctx:IselContext, x:UOp) -> UOp:
  base, off_lo = x.src
  off_hi = off_lo >> 31
  base_lo, base_hi = [UOp(Ops.GETTUPLE, dtypes.uint32, (base,), i) for i in range(2)]
  add_lo = UOp(Ops.INS, dtypes.uint32, (base_lo, off_lo), RDNA3Ops.S_ADD_U32)
  scc_use = UOp(Ops.GETTUPLE, add_lo.dtype, (add_lo,), 0)
  add_hi = UOp(Ops.INS, dtypes.uint32, (base_hi, off_hi, scc_use), RDNA3Ops.S_ADDC_U32)
  pack = UOp(Ops.TUPLE, dtypes.uint64, (add_lo, add_hi), tag=vreg_sgpr(ctx, x))
  return pack

sop_matcher = PatternMatcher([
  # INDEX is lowered to 32bit add pair, this only matches when nothing in the address can be folded into the load/store
  (UPat().index(UPat()).named("x"), lower_index),
  # SOP1
  # this is a noop really
  (UPat(dtype=dtypes.bool).cast(dtypes.int32s, name="x"), partial(lower_sop1, RDNA3Ops.S_MOV_B32)),
  (UPat(dtype=dtypes.int8).cast(dtypes.int32, name="x"), partial(lower_sop1, RDNA3Ops.S_SEXT_I32_I8)),
  (UPat(dtype=dtypes.int16).cast(dtypes.int32, name="x"), partial(lower_sop1, RDNA3Ops.S_SEXT_I32_I16)),
  # SOP2
  ((UPat.var("a", dtypes.int32s).alu(Ops.SUB, UPat.var("b"))).named("x"), partial(lower_sop2, RDNA3Ops.S_SUB_U32)),
  ((UPat.var("a", dtypes.int32s) + UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_ADD_U32)),
  ((UPat.var("a", dtypes.int32s) * UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_MUL_I32)),
  ((UPat.var("a", dtypes.int32s) << UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHL_B32)),
  ((UPat.var("a", dtypes.int64s) << UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHL_B64)),
  ((UPat.var("a", dtypes.uint32) >> UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHR_B32)),
  ((UPat.var("a", dtypes.uint64) >> UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_LSHR_B64)),
  ((UPat.var("a", dtypes.int32) >> UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_ASHR_I32)),
  ((UPat.var("a", dtypes.int64) >> UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_ASHR_I64)),
  ((UPat.var("a", dtypes.int32s+(dtypes.bool,)) & UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_AND_B32)),
  ((UPat.var("a", dtypes.int32s+(dtypes.bool,)) ^ UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_XOR_B32)),
  ((UPat.var("a", dtypes.int32s+(dtypes.bool,)) | UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_OR_B32)),
  ((UPat.var("a", dtypes.int64s) & UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_AND_B64)),
  ((UPat.var("a", dtypes.int64s) ^ UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_XOR_B64)),
  ((UPat.var("a", dtypes.int64s) | UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_OR_B64)),
  ((UPat.var("a", dtypes.int32) < UPat.var("b")).where(UPat.var("b"), UPat.var("a")).named("x"), partial(lower_sop2, RDNA3Ops.S_MAX_I32)),
  ((UPat.var("a", dtypes.uint32) < UPat.var("b")).where(UPat.var("b"), UPat.var("a")).named("x"), partial(lower_sop2, RDNA3Ops.S_MAX_U32)),
  ((UPat.var("a", dtypes.int32) < UPat.var("b")).where(UPat.var("a"), UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_MIN_I32)),
  ((UPat.var("a", dtypes.uint32) < UPat.var("b")).where(UPat.var("a"), UPat.var("b")).named("x"), partial(lower_sop2, RDNA3Ops.S_MIN_U32)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("a", dtypes.int32s), UPat.var("b")).named("x"), partial(lower_s_cselect, RDNA3Ops.S_CSELECT_B32)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("a", dtypes.int64s), UPat.var("b")).named("x"), partial(lower_s_cselect, RDNA3Ops.S_CSELECT_B64)),
  # SOPC
  # comparisons whose user doesn't use the flag, move flag result to register
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LT_I32)),
  (UPat(Ops.CMPLT, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LT_U32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_EQ_I32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_EQ_U32)),
  (UPat(Ops.CMPEQ, dtypes.bool, (UPat(dtype=dtypes.int64s), UPat()), name="x"), partial(lower_sopc_64bit, RDNA3Ops.S_CMP_EQ_U64)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.int32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LG_I32)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.uint32), UPat()), name="x"), partial(lower_sopc, RDNA3Ops.S_CMP_LG_U32)),
  (UPat(Ops.CMPNE, dtypes.bool, (UPat(dtype=dtypes.int64s), UPat()), name="x"), partial(lower_sopc_64bit, RDNA3Ops.S_CMP_LG_U64)),
  # SMEM
  (UPat(Ops.LOAD, src=(UPat(),), name="x"), lower_smem),
])

# TODO: not quite right and some missing cases?
def lower_const(ctx:IselContext, x:UOp) -> UOp|None:
  if x.tag is True: return None
  # if the user needs x in a vgpr move the const to vgpr directly
  if x.tag == "vgpr":
    ret = x.rtag(None)
    return ret.ins(RDNA3Ops.V_MOV_B32, src=(imm(ret.dtype, ret.arg), exec_use))
  if x.dtype in dtypes.ints+(dtypes.bool,) and x.dtype.itemsize <= 32:
    # use s_movk_i32 if value is equivalent to the sign extension of its lower 16bits
    if (x.arg << 16) >> 16 == x.arg: return x.ins(RDNA3Ops.S_MOVK_I32, src=(imm(x.dtype, x.arg),))
  if x.dtype.itemsize <= 32:
    # fallback to s_mov_b32 otherwise
    return x.ins(RDNA3Ops.S_MOV_B32, src=(imm(x.dtype, x.arg),))
  # otherwise need to emit two instructions
  # TODO: could emit shorter encodings if each half fits in 16bits, call this function again in that case
  bits = struct.unpack('Q', struct.pack(unwrap(x.dtype.fmt), x.arg))[0]
  return UOp(Ops.TUPLE, x.dtype, (x.ins(RDNA3Ops.S_MOV_B32, dtype=dtypes.uint32, src=(imm(dtypes.uint32, lo32(bits)))),
                                  x.ins(RDNA3Ops.S_MOV_B32, dtype=dtypes.uint32, src=(imm(dtypes.uint32, hi32(bits))))), tag=vreg_sgpr(ctx, x))

def lower_uniform(ctx:IselContext, x:UOp) -> UOp|None:
  x = cast(UOp|None, sop_matcher.rewrite(x, ctx)) if is_uniform(x) else None
  if x is None: return None
  # always clean the tag to avoid duplicates
  ret = x.rtag(None)
  # if the user requires x to be in a vgpr we insert a move
  # TODO: this should be COPY
  if x.tag == "vgpr": ret = ret.ins(RDNA3Ops.V_MOV_B32, src=(ret, exec_use))
  # if the user requires x to be a mask we convert from scalar bool to mask
  elif x.tag == "vcc":
    assert x.dtype is dtypes.bool
    #ret = ret.where(-1, 0)
    ret = ret.src[0].ins(RDNA3Ops.S_CSELECT_B32, src=(imm(ret.src[0].dtype, -1), imm(ret.src[0].dtype, 0), ret), tag=(ctx.vreg((VCC_LO,) + sz_to_sgpr[x.src[0].dtype.bitsize]),))
  return ret

# TODO: this should probably be done at the end
# here x lowers to a VOP so insert a move from vgpr to sgpr if the user requires x to be in an sgpr
def move_to_sgpr(x:UOp) -> UOp|None:
  if x.tag is None: return None
  # always clean the tag to avoid duplicates
  ret = x.rtag(None)
  if x.tag == "sgpr":
    # if x lowers to VOPC we need to convert the mask to scalar bool
    if x.op in GroupOp.Comparison: ret = ret.ins(RDNA3Ops.S_CMP_LG_U32, src=(ret, imm(x.dtype, 0)))
    else: ret = ret.ins(RDNA3Ops.V_READFIRSTLANE_B32, src=(ret,))
  return ret

def lower_vop1(op:RDNA3Ops, ctx:IselContext, x:UOp) -> UOp:
  if is_vop3(ctx, x): return lower_vop3(op, ctx, x)
  return x.ins(op, src=x.src + (exec_use,))

isel_matcher = PatternMatcher([
  # some constant need to be moved to registers
  (UPat.cvar("x"), lower_const),
  (UPat(GroupOp.Elementwise | {Ops.LOAD, Ops.INDEX}, name="x"), lower_uniform),
  (UPat(GroupOp.Elementwise | {Ops.LOAD}, name="x"), move_to_sgpr),
  # VOP1
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
  # this is a noop really
  (UPat(dtype=dtypes.bool).cast(dtypes.int32s, name="x"), partial(lower_vop1, RDNA3Ops.V_MOV_B32)),
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
])

# in VOP2 only src0 can be a scalar (sgpr/inline const/literal), otherwise the longer VOP3 encoding must be used and we want to avoid that
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
  return x.ins(op, src=vop2_src(a, b, cm) + (exec_use,))

def lower_v_mul_24(op:RDNA3Ops, ctx:IselContext, x:UOp):
  if not all(fits_in_bits(s, 24) for s in x.src): return None
  return lower_vop2(op, ctx, x)

def lower_fabs(op:RDNA3Ops, ctx:IselContext, x:UOp): return lower_vop2(op, ctx, x, x.src[0], imm(dtypes.uint32, 0x7fffffff), True)
def lower_fneg(op:RDNA3Ops, ctx:IselContext, x:UOp): return lower_vop2(op, ctx, x, x.src[0], imm(dtypes.uint32, 0x80000000), True)

# VOP2
isel_matcher += PatternMatcher([
  #(UPat(dtype=dtypes.float32).abs().named("x"), partial(lower_fabs, RDNA3Ops.V_AND_B32)),
  (UPat(dtype=dtypes.float32).alu(Ops.NEG).named("x"), partial(lower_fneg, RDNA3Ops.V_XOR_B32)),
  ((UPat(dtype=dtypes.int32s+(dtypes.bool,)) & UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_AND_B32)),
  ((UPat(dtype=dtypes.int32s+(dtypes.bool,)) ^ UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_XOR_B32)),
  ((UPat(dtype=dtypes.int32s+(dtypes.bool,)) | UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_OR_B32)),
  ((UPat(dtype=dtypes.int32s) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_NC_U32)),
  ((UPat(dtype=dtypes.float16) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_F16)),
  ((UPat(dtype=dtypes.float32) + UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_ADD_F32)),
  ((UPat(dtype=dtypes.float16) * UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_MUL_F16)),
  ((UPat(dtype=dtypes.float32) * UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_MUL_F32)),
  ((UPat(dtype=dtypes.int32) * UPat()).named("x"), partial(lower_v_mul_24, RDNA3Ops.V_MUL_I32_I24)),
  ((UPat(dtype=dtypes.uint32) * UPat()).named("x"), partial(lower_v_mul_24, RDNA3Ops.V_MUL_U32_U24)),
  (UPat.var("b", dtypes.int32s).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_NC_U32)),
  (UPat.var("b", dtypes.float16).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_F16)),
  (UPat.var("b", dtypes.float32).alu(Ops.SUB, UPat.cvar("a")).named("x"), partial(lower_vop2, RDNA3Ops.V_SUBREV_F32)),
  (UPat(dtype=dtypes.int32s).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_NC_U32)),
  (UPat(dtype=dtypes.float16).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_F16)),
  (UPat(dtype=dtypes.float32).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop2, RDNA3Ops.V_SUB_F32)),
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
])

def lower_vopc(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp, b:UOp) -> UOp:
  # this means x's user expects its result to be a scalar boolean and not a mask, so we convert
  if is_vop3(ctx, x): return lower_vop3(op, ctx, x, a, b)
  # VOPC instructions write to VCC but their VOP3 version can write to any SGPR, we fallback to SGPR when VCC isn't available
  # this way we avoid having to rematerialize and let regalloc deal with it
  return x.ins(op, src=vop2_src(a, b, x.op in GroupOp.Commutative) + (exec_use,), tag=(ctx.vreg((VCC_LO,) + sz_to_sgpr[x.src[0].dtype.bitsize]),))

def vopc(op): return partial(lower_vopc, op)

# VOPC
isel_matcher += PatternMatcher([
  (UPat(Ops.CMPLT, src=(UPat.var("b", dtypes.float32), UPat.cvar("a")), name="x"), vopc(RDNA3Ops.V_CMP_GT_F32)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_F16)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_F32)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_F64)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.float16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_F16)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_F32)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_F64)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NEQ_F16)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NEQ_F32)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.float64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NEQ_F64)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_I16)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_I32)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.int64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_I64)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.uint16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_U16)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.uint32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_U32)),
  (UPat(Ops.CMPLT, src=(UPat.var("a", dtypes.uint64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_LT_U64)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.int16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_I16)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.int32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_I32)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.int64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_I64)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.uint16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_U16)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.uint32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_U32)),
  (UPat(Ops.CMPEQ, src=(UPat.var("a", dtypes.uint64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_EQ_U64)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.int16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_I16)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.int32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_I32)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.int64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_I64)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.uint16), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_U16)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.uint32), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_U32)),
  (UPat(Ops.CMPNE, src=(UPat.var("a", dtypes.uint64), UPat.var("b")), name="x"), vopc(RDNA3Ops.V_CMP_NE_U64)),
])

def is_vop3(ctx:IselContext, x:UOp):
  # TODO: include ABS, OMOD and CLMP modifiers
  return x.dtype in dtypes.floats and any(s.op is Ops.NEG and ctx.is_foldable(x, s) for s in x.src)

def lower_vop3(op:RDNA3Ops, ctx:IselContext, x:UOp, *src:UOp) -> UOp:
  if not src: src = x.src
  #src = tuple(s.replace(op=Ops.NOOP, arg=Ops.NEG) if s.op is Ops.NEG and ctx.is_foldable(x, s) else s for s in ssrcs(src))
  src = ssrcs(src)
  return x.ins(op, src=src + (exec_use,))

def lower_vop3_bin(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp, b:UOp) -> UOp: return lower_vop3(op, ctx, x, a, b)
def lower_v_cndmask(op:RDNA3Ops, ctx:IselContext, x:UOp, a:UOp, b:UOp, c:UOp) -> UOp: return lower_vop3(op, ctx, x, a, b, c.rtag("vcc"))

# VOP3
isel_matcher += PatternMatcher([
  (UPat(dtype=dtypes.int16s).alu(Ops.SUB, UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_SUB_NC_U16)),
  ((UPat(dtype=dtypes.int16s) + UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_ADD_NC_U16)),
  ((UPat(dtype=dtypes.float64) + UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_ADD_F64)),
  ((UPat(dtype=dtypes.float64) * UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_MUL_F64)),
  ((UPat(dtype=dtypes.int16s) * UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_MUL_LO_U16)),
  ((UPat(dtype=dtypes.int32s) * UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_MUL_LO_U32)),
  ((UPat(dtype=dtypes.int16s) & UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_AND_B16)),
  ((UPat(dtype=dtypes.int16s) ^ UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_XOR_B16)),
  ((UPat(dtype=dtypes.int16s) | UPat()).named("x"), partial(lower_vop3, RDNA3Ops.V_OR_B16)),
  ((UPat.var("b", dtypes.int16s) << UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_LSHLREV_B16)),
  ((UPat.var("b", dtypes.int64s) << UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_LSHLREV_B64)),
  ((UPat.var("b", dtypes.uint16) >> UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_LSHRREV_B16)),
  ((UPat.var("b", dtypes.uint64) >> UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_LSHRREV_B64)),
  ((UPat.var("b", dtypes.int16) >> UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_ASHRREV_I16)),
  ((UPat.var("b", dtypes.int64) >> UPat.var("a")).named("x"), partial(lower_vop3_bin, RDNA3Ops.V_ASHRREV_I64)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.int16), UPat.var("a")), partial(lower_vop3_bin, RDNA3Ops.V_MAX_I16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.uint16), UPat.var("a")), partial(lower_vop3_bin, RDNA3Ops.V_MAX_U16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("b", dtypes.float64), UPat.var("a")), partial(lower_vop3_bin, RDNA3Ops.V_MAX_F64)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.int16), UPat.var("b")), partial(lower_vop3_bin, RDNA3Ops.V_MIN_I16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.uint16), UPat.var("b")), partial(lower_vop3_bin, RDNA3Ops.V_MIN_U16)),
  ((UPat.var("a") < UPat.var("b")).where(UPat.var("a", dtypes.float64), UPat.var("b")), partial(lower_vop3_bin, RDNA3Ops.V_MIN_F64)),
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("b", dtypes.int16s+(dtypes.float16,)), UPat.var("a")).named("x"),
   partial(lower_v_cndmask, RDNA3Ops.V_CNDMASK_B16)),
  # NOTE: this is VOP2 but you want to lower as VOP3 as it can have 2 CONST srcs and isn't commutative
  (UPat(GroupOp.Comparison, name="c").where(UPat.var("b", dtypes.int32s+(dtypes.float32,)), UPat.var("a")).named("x"),
   partial(lower_v_cndmask, RDNA3Ops.V_CNDMASK_B32)),
  ((UPat.var("a", dtypes.uint32).cast(dtypes.uint64) * UPat.var("b", dtypes.uint32).cast() >> 32).cast(dtypes.uint32).named("x"),
   partial(lower_vop3_bin, RDNA3Ops.V_MUL_HI_U32)),
])

def lower_fma(ctx:IselContext, x:UOp, a:UOp, b:UOp):
  if not ctx.is_foldable(x, a): return None
  if x.dtype is dtypes.float16:
    if x.dtype.count == 2: return x.ins(RDNA3Ops.V_PK_FMAC_F16, src=(*a.src, b))
    if (i:=to_literal(b)) is not None: return x.ins(RDNA3Ops.V_FMAAK_F16, src=(*a.src, i))
    if (i:=to_literal(a.src[1])) is not None: return x.ins(RDNA3Ops.V_FMAMK_F16, src=(a.src[0], i, b))
    return lower_vop3(RDNA3Ops.V_FMAC_F16, ctx, x, *a.src, b)
  if x.dtype is dtypes.float32:
    if (i:=to_literal(b)) is not None: return x.ins(RDNA3Ops.V_FMAAK_F32, src=(*a.src, i))
    if (i:=to_literal(a.src[1])) is not None: return x.ins(RDNA3Ops.V_FMAMK_F32, src=(a.src[0], b, i))
    return lower_vop3(RDNA3Ops.V_FMAC_F32, ctx, x, *a.src, b)
  if x.dtype is dtypes.float64: return lower_vop3(RDNA3Ops.V_FMA_F64, ctx, x.replace(src=(*a.src, b)))
  return None

isel_matcher += PatternMatcher([((UPat(Ops.MUL, dtypes.floats, name="a") + UPat.var("b")).named("x"), lower_fma)])

# [vgpr(u32) + const(u16)]
def lds_address(x:UOp) -> tuple[UOp, UOp]:
  base, idx = x.src
  scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  # full lds address
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST:
    addr = idx.src[0] * scale
    offset = imm(idx.src[1].dtype, idx.src[1].arg * scale)
    # addr must be in a vgpr
    if fits_in_bits(addr, 32, False) and fits_in_bits(offset, 16, False): return (base, addr.rtag("vgpr"), offset)
  # with vgpr 0
  if idx.op is Ops.CONST:
    addr = idx.ins(RDNA3Ops.V_MOV_B32, src=(imm(idx.dtype, 0), exec_use))
    offset = imm(idx.dtype, idx.arg * scale)
    if fits_in_bits(offset, 16, False): return (base, addr, offset)
  # with const 0
  addr = idx * scale
  return (base, addr.rtag("vgpr"), imm(idx.dtype, 0))

# mem_addr = SCRATCH_BASEU64 + SWIZZLE(SGPR_offsetU32 + INST_OFFSETI13, ThreadID)
def scratch_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  base, idx = x.src
  scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  if idx.op is Ops.CONST:
    offset = imm(idx.dtype, idx.arg * scale)
    if fits_in_bits(offset, 13, True): return (base, UOp(Ops.NOOP), imm(dtypes.int32, 0))

  return (x, UOp(Ops.NOOP), imm(dtypes.int32, 0))

# GV:  [vgpr pair(u64) + const(i13)]
# GVS: [sgpr pair(u64) + vgpr(u32) + const(i13)]
# GT:  [sgpr pair(u64) + const(i13) + threadid * 4]
def global_address(x:UOp) -> tuple[UOp, UOp, UOp]:
  base, idx = x.src
  scale = base.dtype.itemsize if isinstance(base.dtype, PtrDType) else 1
  # full GVS address
  if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST:
    addr = idx.src[0] * scale
    offset = imm(idx.src[1].dtype, idx.src[1].arg * scale)
    # addr must be in a vgpr
    if fits_in_bits(addr, 32, False) and fits_in_bits(offset, 13, True): return (base, addr.rtag("vgpr"), offset)

  # GVS with vgpr 0
  vgpr0 = idx.ins(RDNA3Ops.V_MOV_B32, src=(imm(idx.dtype, 0), exec_use))
  if idx.op is Ops.CONST:
    offset = imm(idx.dtype, idx.arg * scale)
    if fits_in_bits(offset, 13, True): return (base, vgpr0, offset)
  
  # try GVS address with 0 offset
  off0 = imm(dtypes.int32, 0)
  addr = idx * scale
  if fits_in_bits(addr, 32, False): return (base, addr.rtag("vgpr"), off0)

  return (x, vgpr0, off0)

def lower_load(ctx:IselContext, x:UOp, alt:UOp|None=None, m:UOp|None=None):
  addrspace, sz = x.src[0].ptrdtype.addrspace, x.dtype.itemsize * 8 if not isinstance(x.dtype, PtrDType) else 64
  # NOTE: RDNA3Ops.****_LOAD_U8 is for bools, int8s are handled by load(int8).cast(int32)
  if addrspace == AddrSpace.LOCAL:
    # TODO: ds_load_2addr_b32/64
    op = {8: RDNA3Ops.DS_LOAD_U8, 32: RDNA3Ops.DS_LOAD_B32, 64: RDNA3Ops.DS_LOAD_B64,
          96: RDNA3Ops.DS_LOAD_B96, 128: RDNA3Ops.DS_LOAD_B128}[sz]
    src = lds_address(x.src[0])
  elif addrspace == AddrSpace.PRIVATE:
    op = {8: RDNA3Ops.SCRATCH_LOAD_U8, 32: RDNA3Ops.SCRATCH_LOAD_B32, 64: RDNA3Ops.SCRATCH_LOAD_B64,
          96: RDNA3Ops.SCRATCH_LOAD_B96, 128: RDNA3Ops.SCRATCH_LOAD_B128}[sz]
    src = scratch_address(x.src[0])
  elif addrspace == AddrSpace.GLOBAL:
    op = {8: RDNA3Ops.GLOBAL_LOAD_U8, 32: RDNA3Ops.GLOBAL_LOAD_B32, 64: RDNA3Ops.GLOBAL_LOAD_B64,
          96: RDNA3Ops.GLOBAL_LOAD_B96, 128: RDNA3Ops.GLOBAL_LOAD_B128}[sz]
    src = global_address(x.src[0])
  else: raise RuntimeError("invalid addrspace")
  # TODO: I tried modeling masked loads as two address so x and alt would share the same register but it doesn't work
  # because of exec mask clobbers, this approach would work if the scheduler constrained alt to be in the same block as x
  # otherwise the right lowering is to lower masked load/store to branches before isel and then lower to exec mask when the branch condition is divergent
  # this would also require a phi node
  # the current approach inserts a cmove after the load, this doesn't work if the load is > 32bits so for now might want to disallow that
  if m is not None:
    assert alt is not None
    m = m.rtag("vcc")
    mask = m.ins(RDNA3Ops.S_AND_B32, src=(m, exec_use), tag=(EXEC_LO, SCC))
    load = x.ins(op, src=src + (mask,), tag=vreg_vgpr(ctx, x))
    return x.ins(RDNA3Ops.V_CNDMASK_B32, src=(alt, load, m, exec_use), tag=vreg_vgpr(ctx, x))
  return x.ins(op, src=src + (exec_use,), tag=vreg_vgpr(ctx, x))

def lower_store(x:UOp, m:UOp|None=None) -> UOp:
  addrspace, sz = x.src[0].ptrdtype.addrspace, x.src[1].dtype.itemsize * 8
  if addrspace == AddrSpace.LOCAL:
    op = {8: RDNA3Ops.DS_STORE_B8, 16: RDNA3Ops.DS_STORE_B16, 32: RDNA3Ops.DS_STORE_B32, 64: RDNA3Ops.DS_STORE_B64,
          96: RDNA3Ops.DS_STORE_B96, 128: RDNA3Ops.DS_STORE_B128}[sz]
    src = lds_address(x.src[0])
  elif addrspace == AddrSpace.PRIVATE:
    op = {8: RDNA3Ops.SCRATCH_STORE_B8, 16: RDNA3Ops.SCRATCH_STORE_B16, 32: RDNA3Ops.SCRATCH_STORE_B32, 64: RDNA3Ops.SCRATCH_STORE_B64,
          96: RDNA3Ops.SCRATCH_STORE_B96, 128: RDNA3Ops.SCRATCH_STORE_B128}[sz]
    src = scratch_address(x.src[0])
  elif addrspace == AddrSpace.GLOBAL:
    op = {8: RDNA3Ops.GLOBAL_STORE_B8, 16: RDNA3Ops.GLOBAL_STORE_B16, 32: RDNA3Ops.GLOBAL_STORE_B32, 64: RDNA3Ops.GLOBAL_STORE_B64,
          96: RDNA3Ops.GLOBAL_STORE_B96, 128: RDNA3Ops.GLOBAL_STORE_B128}[sz]
    src = global_address(x.src[0])
  else: raise RuntimeError("invalid addrspace")
  # stored value needs to be a vgpr
  v = x.src[1].rtag("vgpr")
  mask = exec_use if m is None else m.ins(RDNA3Ops.S_AND_B32, src=(m, exec_use), tag=(EXEC_LO, SCC))
  return x.ins(op, src=src + (v, mask))

isel_matcher += PatternMatcher([
  (UPat(Ops.LOAD, src=(UPat(),), name="x"), lower_load),
  (UPat(Ops.LOAD, src=(UPat(), UPat.var("alt"), UPat.var("m")), name="x"), lower_load),
  (UPat(Ops.STORE, src=(UPat(), UPat()), name="x"), lower_store),
  (UPat(Ops.STORE, src=(UPat(), UPat(), UPat.var("m")), name="x"), lower_store)
])

def lower_special(ctx:IselContext, x:UOp) -> UOp|None:
  if x.src[0].tag: return None
  # the constant in SPECIAL shouldn't be loaded into a register
  x = x.replace(src=(x.src[0].rtag(),))
  # TODO: if lidx1/2 don't exist we can access lidx0 from v0 directly
  if x.arg.startswith("lidx"): return x.ins(RDNA3Ops.V_BFE_U32, src=(x, imm(x.dtype, int(x.arg[-1]) * 10), imm(x.dtype, 10)), tag=vreg_vgpr(ctx, x))
  # NOTE: this assumes if gidx1 exists then gidx0 exists
  if x.arg.startswith("gidx"): return x.rtag((ctx.vreg(S32[2+int(x.arg[-1])]),))
  raise RuntimeError("invalid arg in SPECIAL")

def alloc_vregs(ctx:IselContext, x:UOp) -> UOp|None:
  # skip if already assigned or x doesn't define a register
  if reg_defs(x) or x.dtype is dtypes.void: return None
  if x.arg is RDNA3Ops.V_READFIRSTLANE_B32: return x.rtag(vreg_sgpr(ctx, x))
  if x.arg.name.startswith("V_"): return x.rtag(vreg_vgpr(ctx, x))
  if x.arg.name.startswith("S_CMP"): return x.rtag((SCC,))
  if x.arg.name.startswith("S_"): return x.rtag(vreg_sgpr(ctx, x) + ((SCC,) if x.arg in RDNA3GroupOp.WriteSCC else ()))
  raise RuntimeError(f"can't allocate vreg to {x.arg}")

# STACK becomes TUPLE. TUPLE causes the src live ranges to be coalesced into a single live range
# which causes the src virtuals to be assigned as a single contiguous virtual in regalloc
# this requires that each src defines its own virtual so that each gets a slice
# so each repeated src needs to be deduped by assigning a new virtual, i.e. src=(v0, v1, v1, v2) is invalid as v1 can't be assigned 2 diff regs
# TODO: if a src has multiple diff users need to insert a move instruction between that src and x
def lower_tuple(ctx:IselContext, x:UOp) -> UOp:
  # if user requires x to be in a vgpr don't use sgpr otherwise you eat several moves, instead use gpr directly
  # TODO: this still generates too many moves if one user tags "vgpr" and another doesn't but allows vgpr
  to_sgpr = is_uniform(x) and x.tag != "vgpr"
  src = []
  for s in x.src:
    # we deliberately assign virtuals here, this is because each src of x will be assigned a unique register so each src needs its own virtual
    if not ctx.is_foldable(x, s):
      s = s.ins(RDNA3Ops.S_MOV_B32, src=(s,), tag=vreg_sgpr(ctx, s)) if to_sgpr else s.ins(RDNA3Ops.V_MOV_B32, src=(s, exec_use), tag=vreg_vgpr(ctx, s))
    # if x goes to vgpr we need to tag every src as needing vgpr, and vice versa for sgpr
    else: s = s.rtag("sgpr" if to_sgpr else "vgpr")
    src.append(s)
  return x.replace(op=Ops.TUPLE, src=tuple(src), tag=vreg_sgpr(ctx, x) if to_sgpr else vreg_vgpr(ctx, x))

def lower_gettuple(x:UOp) -> UOp:
  ret = x.replace(op=Ops.GETTUPLE, arg=x.arg[0])
  if is_uniform(x) and x.tag == "vgpr": ret = ret.ins(RDNA3Ops.V_MOV_B32, src=(ret, exec_use))
  return ret

# extra
isel_matcher += PatternMatcher([
  # TODO: handle divergent loops
  # range is lowered to acc, cmp, jmp after regalloc
  (UPat(Ops.RANGE, src=(UPat.cvar("c"),), allow_any_len=True, name="x"), lambda c,x: x.replace(src=(imm(c.dtype, c.arg),) + x.src[1:])),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: x.replace(tag=vreg_sgpr(ctx, x)) if not isinstance(x.tag, tuple) else None),
  # kernel arguments are passed in a buffer whose pointer is in s[0:1], we need to load from that pointer to get the arguments in registers
  # TODO: this is a hack as each Ops.PARAM defines the same s[0:1], in reality a single node should define that and Ops.PARAM should disappear
  # into a load from s[0:1] + offset. But stuff after isel relies on Ops.PARAM so it needs to stay in the graph
  (UPat(Ops.PARAM, name="x"), lambda x: x.rtag().index(UOp.const(dtypes.int32, x.arg.slot), ptr=True).load(dtype=x.dtype) if not x.tag else None),
  # NOTE: def_reg here hold the live in values is here to hold s[0:1] so nothing else uses it, this is because the kernel args are passed in a buffer pointed to by s[0:1]
  # s[0:1] holds the pointer to the buffer in which the kernel args are passed in
  # v0 holds the workitem id
  (UPat(Ops.SINK, name="x"), lambda x:
   x.replace(src=(x.ins(RDNA3Ops.S_ENDPGM, src=x.src + (def_reg(dtypes.uint64, S64[0]), def_reg(dtypes.uint32, V32[0]))),)) if not x.src or x.src[0].arg is not RDNA3Ops.S_ENDPGM else None),
  (UPat(Ops.SPECIAL, name="x"), lower_special),
  (UPat(Ops.STACK, name="x"), lower_tuple),
  # GEP becomes GETTUPLE
  (UPat(Ops.LOAD).gep(name="x"), lower_gettuple),
  # assign vregs to ops that don't have special constraints
  (UPat(Ops.INS, name="x"), alloc_vregs),
])

# ***** post instruction selection *****

# insert moves from vgpr to sgpr when the src of a SOP instruction is a vgpr
def vgpr_to_sgpr(x:UOp) -> UOp|None:
  if not x.arg.name.startswith("S_") or x.dtype is dtypes.void: return None
  src = tuple(s.ins(RDNA3Ops.V_READFIRSTLANE_B32, src=(s,), tag=None) if (r:=reg_use(s)) is not None and is_vgpr(r.cons[0]) else s for s in x.src)
  assert src == x.src
  return x.replace(src=src)

post_isel_matcher = PatternMatcher([
  (UPat(Ops.INS, name="x"), vgpr_to_sgpr),
  (UPat(Ops.INS, name="x"), alloc_vregs),
])

# ***** pre register allocation *****

def flag_rematerialize(ctx:FlagRematContext, x:UOp) -> tuple[UOp, list[UOp]]:
  flags = (SCC, EXEC_LO)
  remats = []
  for s in x.src:
    reg = reg_use(s)
    # if s defines a flag and it's not in ctx.flags it's because its been clobbered so it needs to be rematerialized
    if reg in flags and ctx.flags[reg] is not s: remats.append(s)

  for u in remats + [x]:
    for reg in reg_defs(u):
      if reg in flags: ctx.flags[reg] = u
  # HACK
  if x.op in (Ops.END, Ops.RANGE): ctx.flags[SCC] = x

  # exec_use isn't an instruction just the initial exec state, however to remat we need to emit an instruction that resets the exec state
  remats = [u if u is not exec_use else u.ins(RDNA3Ops.S_MOV_B32, src=(imm(u.dtype, -1),)) for u in remats]

  return (x, remats + [x])

flag_remat_matcher = PatternMatcher([(UPat((Ops.DEFINE_REG, Ops.INS, Ops.RANGE, Ops.END), name="x"), flag_rematerialize)])

# ***** post register allocation *****

# TODO: control flow should be overhauled so that this isn't necessary
def lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = "_".join(str(i) for i in x.arg[:-1])
  acc = x.ins(RDNA3Ops.S_MOVK_I32, src=(imm(x.dtype, 0),) + x.src[1:])
  label = UOp(Ops.INS, arg=RDNA3Ops.LABEL, tag=f".LOOP_{loop_label}")
  cmp = UOp(Ops.INS, arg=RDNA3Ops.S_CMP_LT_U32, src=(acc, x.src[0]), tag=(SCC,))
  jump_out = UOp(Ops.INS, arg=RDNA3Ops.S_CBRANCH_SCC0, src=(cmp,), tag=f".LOOP_OUT_{loop_label}")
  ctx.loop_label[acc] = loop_label
  return (acc, [acc, label, cmp, jump_out])

def two_address(x:UOp) -> int:
  # masked loads are modelled as 2 address where the alt value is the reused src
  if isinstance(x.arg, RDNA3Ops) and x.arg.name.startswith(("SCRATCH_LOAD", "GLOBAL_LOAD", "DS_LOAD")) and len(x.src) == 5: return 3
  return 2 if x.arg in RDNA3GroupOp.TwoAddress else -1

def lower_two_address(x:UOp) -> tuple[UOp, list[UOp]]|None:
  if two_address(x) == -1: return None
  reused_src = x.src[two_address(x)]
  reg_def = reg_defs(x)[0]
  if reg_use(reused_src) == reg_def: return None
  # instead of inserting a move we downgrade these ops to their VOP3 version
  if x.arg is RDNA3Ops.V_FMAC_F16: return (nx:=x.replace(arg=RDNA3Ops.V_FMA_F16), [nx])
  if x.arg is RDNA3Ops.V_FMAC_F32: return (nx:=x.replace(arg=RDNA3Ops.V_FMA_F32), [nx])
  move = x.ins(RDNA3Ops.V_MOV_B32, src=(reused_src,), tag=(reg_def,))
  return (x, [move, x])

post_regalloc_matcher = PatternMatcher([
  # rewrite RANGE to ACC = 0 -> LABEL -> JUMP if ACC >= loop bound
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: lower_range(ctx, x)),
  # rewrite END to ACC + 1 -> JUMP -> LABEL, also add the out of loop JUMP to the src so this becomes the jump target
  (UPat(Ops.END, name="x"), lambda ctx,x: (jmp:=UOp(Ops.INS, arg=RDNA3Ops.S_BRANCH, tag=f".LOOP_{ctx.loop_label[x.src[1]]}"),
   [x.src[1].ins(RDNA3Ops.S_ADD_U32, src=(x.src[1], imm(x.src[1].dtype, 1))), jmp, UOp(Ops.INS, arg=RDNA3Ops.LABEL, tag=f".LOOP_OUT_{ctx.loop_label[x.src[1]]}")])),
  # HACK: these 2 are hacks
  (UPat(Ops.PARAM, name="x"), lambda x: (nx:=x.rtag((S64[0],)), [nx])),
  (UPat(Ops.SPECIAL, name="x"), lambda x: (nx:=x.rtag((V32[0],)), [nx]) if x.arg.startswith("lidx") else None),
  #(UPat(Ops.INS, name="x"), lower_two_address),
  # these are two address, if regalloc didn't coalesce downgrade them to their long encoding version
  (UPat(Ops.INS, arg=RDNA3Ops.V_FMAC_F16, name="x"), lambda x: (nx:=x.replace(arg=RDNA3Ops.V_FMA_F16), [nx]) if x.reg != x.src[2].reg else None),
  (UPat(Ops.INS, arg=RDNA3Ops.V_FMAC_F32, name="x"), lambda x: (nx:=x.replace(arg=RDNA3Ops.V_FMA_F32), [nx]) if x.reg != x.src[2].reg else None),
])

# ***** post register allocation 2 *****
# this inserts the required s_waitcnt instructions

VMcnt = 0
VScnt = 1
LGKMcnt = 2

def insert_waitcnt(ctx:PostRegallocContext, x:UOp) -> tuple[UOp, list[UOp]]:
  defs, uses, needed, waitcnt = reg_defs(x), reg_uses(x), {}, []

  for reg in defs + uses:
    if reg in ctx.pending:
      (c, s) = ctx.pending[reg]
      needed[c] = min(needed.get(c, float("inf")), ctx.issued[c] - s)

  if needed:
    if VMcnt in needed or LGKMcnt in needed:
      cnt = imm(dtypes.int16, needed.get(VMcnt, 0x3F) << 10 | needed.get(LGKMcnt, 0x3F) << 4 | 0x7)
      waitcnt += [UOp(Ops.INS, dtypes.void, (cnt,), RDNA3Ops.S_WAITCNT)]
    if VScnt in needed:
      waitcnt += [UOp(Ops.INS, dtypes.void, (imm(dtypes.int16, needed[VScnt]),), RDNA3Ops.S_WAITCNT_VSCNT)]
    # clear what's now done
    ctx.pending = {reg:(c,s) for reg,(c,s) in ctx.pending.items() if c not in needed or ctx.issued[c] - s < needed[c]}

  counter = None
  if x.arg.name.startswith(("SCRATCH_LOAD", "GLOBAL_LOAD")): counter = VMcnt
  elif x.arg.name.startswith(("SCRATCH_STORE", "GLOBAL_STORE")): counter = VScnt
  elif x.arg.name.startswith("S_LOAD"): counter = LGKMcnt
  if counter is not None:
    if counter not in ctx.issued: ctx.issued[counter] = 0
    ctx.issued[counter] += 1
    # for loads the def is what's tied, for stores (which don't define a register) it's the uses
    for reg in (defs if defs else uses): ctx.pending[reg] = (counter, ctx.issued[counter])

  return (x, waitcnt + [x])

post_regalloc_matcher2 = PatternMatcher([
  (UPat(Ops.INS, name="x"), insert_waitcnt),
])

# ***** RDNA3 instruction encoding *****

def is_sgpr(r:Register|None) -> bool: return r is not None and r in sz_to_sgpr.get(r.width, ())
def is_vgpr(r:Register|None) -> bool: return r is not None and r in sz_to_vgpr.get(r.width, ())

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
  return unwrap(reg_use(x)).index

def enc_sopk(opc:int, x:UOp) -> bytes:
  simm16 = x.src[0].arg
  sdst = reg.index if (reg:=reg_use(x)) is not None else 124
  enc = struct.pack('<I', 0b1011 << 28 | opc << 23 | sdst << 16 | simm16)
  return enc

def enc_sop1(opc:int, x:UOp) -> bytes:
  ssrc0 = enc_ssrc(x.src[0])
  sdst = enc_ssrc(x)
  enc = struct.pack('<I', 0b101111101 << 23 | sdst << 16 | opc << 8 | ssrc0)
  if ssrc0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  return enc

def enc_sop2(opc:int, x:UOp) -> bytes:
  ssrc0 = enc_ssrc(x.src[0])
  ssrc1 = enc_ssrc(x.src[1])
  sdst = enc_ssrc(x)
  enc = struct.pack('<I', 0b10 << 30 | opc << 23 | sdst << 16 | ssrc1 << 8 | ssrc0)
  if ssrc0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif ssrc1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  return enc

def enc_sopc(opc:int, x:UOp) -> bytes:
  ssrc0 = enc_ssrc(x.src[0])
  ssrc1 = enc_ssrc(x.src[1])
  enc = struct.pack('<I', 0b101111110 << 23 | opc << 16 | ssrc1 << 8 | ssrc0)
  if ssrc0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif ssrc1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  return enc

def enc_sopp(opc:int, x:UOp) -> bytes:
  simm16 = x.src[0].arg if x.src and x.src[0].op is Ops.CONST else 0
  enc = struct.pack('<I', 0b101111111 << 23 | opc << 16 | simm16 & 0xFFFF)
  return enc

def bits(*fields: tuple[int, int]) -> int:
  acc = 0
  for value, width in fields:
      assert -(1 << (width-1)) <= value < (1 << width), f"{value} doesn't fit in {width} bits"
      acc = (acc << width) | (value & ((1 << width) - 1))
  return acc

def enc_smem(opc:int, x:UOp) -> bytes:
  sbase = unwrap(reg_use(x.src[0])).index // 2
  soffset = reg.index if (reg:=reg_use(x.src[1])) is not None else 124
  offset = x.src[2].arg
  sdata = reg.index if (reg:=reg_use(x)) is not None else unwrap(reg_use(x.src[3])).index
  enc = struct.pack('<Q', soffset << 57 | (offset & 0x1FFFFF) << 32 | 0b111101 << 26 | opc << 18 | 0 << 16 | 0 << 14 | sdata << 6 | sbase)
  return enc

def enc_src(x:UOp) -> int:
  if is_vgpr(v:=reg_use(x)): return 256 + v.index
  return enc_ssrc(x)

def enc_vop3(opc:int, x:UOp) -> bytes:
  src0 = enc_src(x.src[0])
  src1 = enc_src(x.src[1]) if len(x.src) > 1 else 0
  # use of EXEC here is always implicit
  src2 = enc_src(x.src[2]) if len(x.src) > 2 and reg_use(x.src[2]) != EXEC_LO else 0
  vdst = unwrap(reg_use(x)).index
  neg = sum(1 << i for i,s in enumerate(x.src) if s.arg is Ops.NEG)
  enc = struct.pack('<Q', neg << 61 | 0 << 59 | src2 << 50 | src1 << 41 | src0 << 32 | 0b110101 << 26 | opc << 16 | 0 << 15 | 0 << 11 | 0 << 8 | vdst)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  elif src1 == 255: enc += struct.pack(unwrap(x.src[1].dtype.fmt), x.src[1].arg)
  elif src2 == 255: enc += struct.pack(unwrap(x.src[2].dtype.fmt), x.src[2].arg)
  return enc

def is_enc_vop3(x:UOp) -> bool: return any(s.arg is Ops.NEG for s in x.src) or len(x.src[:-1]) > 1 and not is_vgpr(reg_use(x.src[1]))

def enc_vop2(opc:int, x:UOp) -> bytes:
  if is_enc_vop3(x): return enc_vop3(opc + 256, x)
  src0 = enc_src(x.src[0])
  vsrc1 = unwrap(reg_use(x.src[1])).index
  vdst = unwrap(reg_use(x)).index
  enc = struct.pack('<I', opc << 25 | vdst << 17 | vsrc1 << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  # this is for FMAAK/FMAMK instructions
  elif len(x.src) == 3 and x.src[2].op is Ops.CONST: enc += struct.pack(unwrap(x.src[2].dtype.fmt), x.src[2].arg)
  return enc

def enc_vopc(opc:int, x:UOp) -> bytes:
  if is_enc_vop3(x) or reg_use(x) != VCC_LO: return enc_vop3(opc, x)
  src0 = enc_src(x.src[0])
  vsrc1 = unwrap(reg_use(x.src[1])).index
  enc = struct.pack('<I', 0b111110 << 25 | opc << 17 | vsrc1 << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  return enc

def enc_vop1(opc:int, x:UOp) -> bytes:
  if is_enc_vop3(x): return enc_vop3(opc + 384, x)
  src0 = enc_src(x.src[0])
  vdst = unwrap(reg_use(x)).index
  enc = struct.pack('<I', 0b111111 << 25 | vdst << 17 | opc << 9 | src0)
  if src0 == 255: enc += struct.pack(unwrap(x.src[0].dtype.fmt), x.src[0].arg)
  #if src0 == 255: enc += struct.pack('<I', x.src[0].arg)
  return enc

def enc_vmem(opc:int, seg:int, x:UOp) -> bytes:
  saddr = reg.index if (reg:=reg_use(x.src[0])) is not None and is_sgpr(reg) else 124
  addr = reg_use(x.src[1 if saddr != 124 else 0])
  if seg != 1: addr, sve = unwrap(addr).index, 0
  elif addr is not None: addr, sve = addr.index, 1
  else: addr, sve = 0, 0
  offset = x.src[2].arg & 0x1FFF
  vdst = reg.index if (reg:=reg_use(x)) is not None else 0
  data = unwrap(reg_use(x.src[3])).index if reg is None else 0
  enc = struct.pack('<Q', vdst << 56 | sve << 55 | saddr << 48 | data << 40 | addr << 32 | 0b110111 << 26 | opc << 18 | seg << 16 | 0 << 15 | 0 << 14 | 0 << 13 | offset)
  return enc

def enc_ds(opc:int, x:UOp) -> bytes:
  offset = x.src[2].arg
  addr = unwrap(reg_use(x.src[1])).index
  vdst = reg.index if (reg:=reg_use(x)) is not None else 0
  data0 = unwrap(reg_use(x.src[3])).index if reg is None else 0
  data1 = 0
  enc = struct.pack('<Q', vdst << 56 | data1 << 48 | data0 << 40 | addr << 32 | 0b110110 << 26 | opc << 18 | 0 << 17 | offset)
  return enc

encodings = {
  # SOPK
  RDNA3Ops.S_MOVK_I32: partial(enc_sopk, 0), RDNA3Ops.S_WAITCNT_VSCNT: partial(enc_sopk, 24),
  # SOP1
  RDNA3Ops.S_SEXT_I32_I8: partial(enc_sop1, 14), RDNA3Ops.S_SEXT_I32_I16: partial(enc_sop1, 15),
  RDNA3Ops.S_MOV_B32: partial(enc_sop1, 0),
  # SOPC
  RDNA3Ops.S_CMP_EQ_I32: partial(enc_sopc, 0), RDNA3Ops.S_CMP_EQ_U32: partial(enc_sopc, 6), RDNA3Ops.S_CMP_EQ_U64: partial(enc_sopc, 16),
  RDNA3Ops.S_CMP_LG_I32: partial(enc_sopc, 1), RDNA3Ops.S_CMP_LG_U32: partial(enc_sopc, 7), RDNA3Ops.S_CMP_LG_U64: partial(enc_sopc, 17),
  RDNA3Ops.S_CMP_LT_I32: partial(enc_sopc, 4), RDNA3Ops.S_CMP_LT_U32: partial(enc_sopc, 10),
  # SOP2
  RDNA3Ops.S_ADD_U32: partial(enc_sop2, 0), RDNA3Ops.S_SUB_U32: partial(enc_sop2, 1), RDNA3Ops.S_MUL_I32: partial(enc_sop2, 44),
  RDNA3Ops.S_AND_B32: partial(enc_sop2, 22), RDNA3Ops.S_AND_B64: partial(enc_sop2, 23), RDNA3Ops.S_OR_B32: partial(enc_sop2, 24),
  RDNA3Ops.S_OR_B64: partial(enc_sop2, 25), RDNA3Ops.S_XOR_B32: partial(enc_sop2, 26), RDNA3Ops.S_XOR_B64: partial(enc_sop2, 27),
  RDNA3Ops.S_LSHL_B32: partial(enc_sop2, 8), RDNA3Ops.S_LSHL_B64: partial(enc_sop2, 9), RDNA3Ops.S_LSHR_B32: partial(enc_sop2, 10),
  RDNA3Ops.S_LSHR_B64: partial(enc_sop2, 11), RDNA3Ops.S_ASHR_I32: partial(enc_sop2, 12), RDNA3Ops.S_ASHR_I64: partial(enc_sop2, 13),
  RDNA3Ops.S_MIN_I32: partial(enc_sop2, 18), RDNA3Ops.S_MIN_U32: partial(enc_sop2, 19),
  RDNA3Ops.S_MAX_I32: partial(enc_sop2, 20), RDNA3Ops.S_MAX_U32: partial(enc_sop2, 21),
  RDNA3Ops.S_CSELECT_B32: partial(enc_sop2, 48), RDNA3Ops.S_CSELECT_B64: partial(enc_sop2, 49),
  RDNA3Ops.S_ADDC_U32: partial(enc_sop2, 4),
  # SOPP
  RDNA3Ops.S_NOP: partial(enc_sopp, 0), RDNA3Ops.S_WAITCNT: partial(enc_sopp, 9),
  RDNA3Ops.S_BRANCH: partial(enc_sopp, 32), RDNA3Ops.S_CBRANCH_SCC0: partial(enc_sopp, 33), RDNA3Ops.S_ENDPGM: partial(enc_sopp, 48),
  # SMEM
  RDNA3Ops.S_LOAD_B32: partial(enc_smem, 0), RDNA3Ops.S_LOAD_B64: partial(enc_smem, 1), RDNA3Ops.S_LOAD_B128: partial(enc_smem, 2),
  RDNA3Ops.S_LOAD_B256: partial(enc_smem, 3), RDNA3Ops.S_LOAD_B512: partial(enc_smem, 4),
  # VOP1
  RDNA3Ops.V_MOV_B32: partial(enc_vop1, 1), RDNA3Ops.V_READFIRSTLANE_B32: partial(enc_vop1, 2),
  RDNA3Ops.V_EXP_F16: partial(enc_vop1, 88), RDNA3Ops.V_EXP_F32: partial(enc_vop1, 37),
  RDNA3Ops.V_LOG_F16: partial(enc_vop1, 87), RDNA3Ops.V_LOG_F32: partial(enc_vop1, 39),
  RDNA3Ops.V_SIN_F16: partial(enc_vop1, 96), RDNA3Ops.V_SIN_F32: partial(enc_vop1, 53),
  RDNA3Ops.V_RCP_F16: partial(enc_vop1, 84), RDNA3Ops.V_RCP_F32: partial(enc_vop1, 42), RDNA3Ops.V_RCP_F64: partial(enc_vop1, 47),
  RDNA3Ops.V_SQRT_F16: partial(enc_vop1, 85), RDNA3Ops.V_SQRT_F32: partial(enc_vop1, 51), RDNA3Ops.V_SQRT_F64: partial(enc_vop1, 52),
  RDNA3Ops.V_TRUNC_F16: partial(enc_vop1, 93), RDNA3Ops.V_TRUNC_F32: partial(enc_vop1, 33), RDNA3Ops.V_TRUNC_F64: partial(enc_vop1, 23),
  RDNA3Ops.V_CVT_F16_U16: partial(enc_vop1, 80), RDNA3Ops.V_CVT_F16_I16: partial(enc_vop1, 81),
  RDNA3Ops.V_CVT_U16_F16: partial(enc_vop1, 82), RDNA3Ops.V_CVT_I16_F16: partial(enc_vop1, 83),
  RDNA3Ops.V_CVT_F32_U32: partial(enc_vop1, 6), RDNA3Ops.V_CVT_F32_I32: partial(enc_vop1, 5),
  RDNA3Ops.V_CVT_U32_F32: partial(enc_vop1, 7), RDNA3Ops.V_CVT_I32_F32: partial(enc_vop1, 8),
  RDNA3Ops.V_CVT_F64_U32: partial(enc_vop1, 22), RDNA3Ops.V_CVT_F64_I32: partial(enc_vop1, 4),
  RDNA3Ops.V_CVT_U32_F64: partial(enc_vop1, 21), RDNA3Ops.V_CVT_I32_F64: partial(enc_vop1, 3),
  RDNA3Ops.V_CVT_F16_F32: partial(enc_vop1, 10), RDNA3Ops.V_CVT_F64_F32: partial(enc_vop1, 16),
  RDNA3Ops.V_CVT_F32_F16: partial(enc_vop1, 11), RDNA3Ops.V_CVT_F32_F64: partial(enc_vop1, 15),
  RDNA3Ops.V_CVT_I32_I16: partial(enc_vop1, 106), RDNA3Ops.V_CVT_U32_U16: partial(enc_vop1, 107),
  # VOP2
  RDNA3Ops.V_CNDMASK_B32: partial(enc_vop2, 1),
  RDNA3Ops.V_ADD_NC_U32: partial(enc_vop2, 37), RDNA3Ops.V_SUB_NC_U32: partial(enc_vop2, 38), RDNA3Ops.V_SUBREV_NC_U32: partial(enc_vop2, 39),
  RDNA3Ops.V_LSHLREV_B32: partial(enc_vop2, 24), RDNA3Ops.V_LSHRREV_B32: partial(enc_vop2, 25), RDNA3Ops.V_ASHRREV_I32: partial(enc_vop2, 26),
  RDNA3Ops.V_MUL_I32_I24: partial(enc_vop2, 9), RDNA3Ops.V_MUL_U32_U24: partial(enc_vop2, 11),
  RDNA3Ops.V_SUBREV_F16: partial(enc_vop2, 52), RDNA3Ops.V_SUBREV_F32: partial(enc_vop2, 5),
  RDNA3Ops.V_ADD_F16: partial(enc_vop2, 50), RDNA3Ops.V_ADD_F32: partial(enc_vop2, 3),
  RDNA3Ops.V_MUL_F16: partial(enc_vop2, 53), RDNA3Ops.V_MUL_F32: partial(enc_vop2, 8),
  RDNA3Ops.V_SUB_F16: partial(enc_vop2, 51), RDNA3Ops.V_SUB_F32: partial(enc_vop2, 4),
  RDNA3Ops.V_MAX_F16: partial(enc_vop2, 57), RDNA3Ops.V_MAX_F32: partial(enc_vop2, 16),
  RDNA3Ops.V_MAX_I32: partial(enc_vop2, 18), RDNA3Ops.V_MAX_U32: partial(enc_vop2, 20),
  RDNA3Ops.V_MIN_F16: partial(enc_vop2, 58), RDNA3Ops.V_MIN_F32: partial(enc_vop2, 15),
  RDNA3Ops.V_MIN_I32: partial(enc_vop2, 17), RDNA3Ops.V_MIN_U32: partial(enc_vop2, 19),
  RDNA3Ops.V_AND_B32: partial(enc_vop2, 27), RDNA3Ops.V_OR_B32: partial(enc_vop2, 28),
  RDNA3Ops.V_XOR_B32: partial(enc_vop2, 29),
  RDNA3Ops.V_FMAC_F16: partial(enc_vop2, 54), RDNA3Ops.V_FMAC_F32: partial(enc_vop2, 43),
  RDNA3Ops.V_FMAMK_F16: partial(enc_vop2, 55), RDNA3Ops.V_FMAMK_F32: partial(enc_vop2, 44),
  RDNA3Ops.V_FMAAK_F16: partial(enc_vop2, 56), RDNA3Ops.V_FMAAK_F32: partial(enc_vop2, 45),
  # VOPC
  RDNA3Ops.V_CMP_GT_F32: partial(enc_vopc, 20),
  RDNA3Ops.V_CMP_LT_F16: partial(enc_vopc, 1), RDNA3Ops.V_CMP_LT_F32: partial(enc_vopc, 17), RDNA3Ops.V_CMP_LT_F64: partial(enc_vopc, 33),
  RDNA3Ops.V_CMP_EQ_F16: partial(enc_vopc, 2), RDNA3Ops.V_CMP_EQ_F32: partial(enc_vopc, 18), RDNA3Ops.V_CMP_EQ_F64: partial(enc_vopc, 34),
  RDNA3Ops.V_CMP_NEQ_F16: partial(enc_vopc, 13), RDNA3Ops.V_CMP_NEQ_F32: partial(enc_vopc, 29), RDNA3Ops.V_CMP_NEQ_F64: partial(enc_vopc, 45),
  RDNA3Ops.V_CMP_LT_I16: partial(enc_vopc, 49), RDNA3Ops.V_CMP_LT_I32: partial(enc_vopc, 65), RDNA3Ops.V_CMP_LT_I64: partial(enc_vopc, 81),
  RDNA3Ops.V_CMP_LT_U16: partial(enc_vopc, 57), RDNA3Ops.V_CMP_LT_U32: partial(enc_vopc, 73), RDNA3Ops.V_CMP_LT_U64: partial(enc_vopc, 89),
  RDNA3Ops.V_CMP_EQ_I16: partial(enc_vopc, 50), RDNA3Ops.V_CMP_EQ_I32: partial(enc_vopc, 66), RDNA3Ops.V_CMP_EQ_I64: partial(enc_vopc, 82),
  RDNA3Ops.V_CMP_EQ_U16: partial(enc_vopc, 58), RDNA3Ops.V_CMP_EQ_U32: partial(enc_vopc, 74), RDNA3Ops.V_CMP_EQ_U64: partial(enc_vopc, 90),
  RDNA3Ops.V_CMP_NE_I16: partial(enc_vopc, 53), RDNA3Ops.V_CMP_NE_I32: partial(enc_vopc, 69), RDNA3Ops.V_CMP_NE_I64: partial(enc_vopc, 85),
  RDNA3Ops.V_CMP_NE_U16: partial(enc_vopc, 61), RDNA3Ops.V_CMP_NE_U32: partial(enc_vopc, 77), RDNA3Ops.V_CMP_NE_U64: partial(enc_vopc, 93),
  # VOP3
  RDNA3Ops.V_FMA_F16: partial(enc_vop3, 584), RDNA3Ops.V_FMA_F32: partial(enc_vop3, 531), RDNA3Ops.V_FMA_F64: partial(enc_vop3, 532),
  RDNA3Ops.V_AND_B16: partial(enc_vop3, 866), RDNA3Ops.V_XOR_B16: partial(enc_vop3, 868), RDNA3Ops.V_OR_B16: partial(enc_vop3, 867),
  RDNA3Ops.V_MAX_I16: partial(enc_vop3, 778), RDNA3Ops.V_MAX_U16: partial(enc_vop3, 777), RDNA3Ops.V_MAX_F64: partial(enc_vop3, 810),
  RDNA3Ops.V_MIN_I16: partial(enc_vop3, 780), RDNA3Ops.V_MIN_U16: partial(enc_vop3, 779), RDNA3Ops.V_MIN_F64: partial(enc_vop3, 809),
  RDNA3Ops.V_LSHLREV_B16: partial(enc_vop3, 824), RDNA3Ops.V_LSHLREV_B64: partial(enc_vop3, 828),
  RDNA3Ops.V_LSHRREV_B16: partial(enc_vop3, 825), RDNA3Ops.V_LSHRREV_B64: partial(enc_vop3, 829),
  RDNA3Ops.V_ASHRREV_I16: partial(enc_vop3, 826), RDNA3Ops.V_ASHRREV_I64: partial(enc_vop3, 830),
  RDNA3Ops.V_MUL_LO_U16: partial(enc_vop3, 773), RDNA3Ops.V_MUL_LO_U32: partial(enc_vop3, 812), RDNA3Ops.V_MUL_HI_U32: partial(enc_vop3, 813),
  RDNA3Ops.V_BFE_U32: partial(enc_vop3, 528),
  RDNA3Ops.V_CNDMASK_B16: partial(enc_vop3, 605),
  # DS
  RDNA3Ops.DS_LOAD_U8: partial(enc_ds, 58), RDNA3Ops.DS_LOAD_I8: partial(enc_ds, 57),
  RDNA3Ops.DS_LOAD_U16: partial(enc_ds, 60), RDNA3Ops.DS_LOAD_I16: partial(enc_ds, 59),
  RDNA3Ops.DS_LOAD_B32: partial(enc_ds, 54), RDNA3Ops.DS_LOAD_B64: partial(enc_ds, 118),
  RDNA3Ops.DS_STORE_B96: partial(enc_ds, 254), RDNA3Ops.DS_STORE_B128: partial(enc_ds, 255),
  RDNA3Ops.DS_STORE_B8: partial(enc_ds, 30), RDNA3Ops.DS_STORE_B16: partial(enc_ds, 31),
  RDNA3Ops.DS_STORE_B32: partial(enc_ds, 13), RDNA3Ops.DS_STORE_B64: partial(enc_ds, 77),
  RDNA3Ops.DS_STORE_B96: partial(enc_ds, 222), RDNA3Ops.DS_STORE_B128: partial(enc_ds, 223),
  # GLOBAL
  RDNA3Ops.GLOBAL_LOAD_U8: partial(enc_vmem, 16, 2), RDNA3Ops.GLOBAL_LOAD_I8: partial(enc_vmem, 17, 2),
  RDNA3Ops.GLOBAL_LOAD_U16: partial(enc_vmem, 18, 2), RDNA3Ops.GLOBAL_LOAD_I16: partial(enc_vmem, 19, 2),
  RDNA3Ops.GLOBAL_LOAD_B32: partial(enc_vmem, 20, 2), RDNA3Ops.GLOBAL_LOAD_B64: partial(enc_vmem, 21, 2),
  RDNA3Ops.GLOBAL_LOAD_B96: partial(enc_vmem, 22, 2), RDNA3Ops.GLOBAL_LOAD_B128: partial(enc_vmem, 23, 2),
  RDNA3Ops.GLOBAL_STORE_B8: partial(enc_vmem, 24, 2), RDNA3Ops.GLOBAL_STORE_B16: partial(enc_vmem, 25, 2),
  RDNA3Ops.GLOBAL_STORE_B32: partial(enc_vmem, 26, 2), RDNA3Ops.GLOBAL_STORE_B64: partial(enc_vmem, 27, 2),
  RDNA3Ops.GLOBAL_STORE_B96: partial(enc_vmem, 28, 2), RDNA3Ops.GLOBAL_STORE_B128: partial(enc_vmem, 29, 2),
  # SCRATCH
  RDNA3Ops.SCRATCH_LOAD_U8: partial(enc_vmem, 16, 1), RDNA3Ops.SCRATCH_LOAD_I8: partial(enc_vmem, 17, 1),
  RDNA3Ops.SCRATCH_LOAD_U16: partial(enc_vmem, 18, 1), RDNA3Ops.SCRATCH_LOAD_I16: partial(enc_vmem, 19, 1),
  RDNA3Ops.SCRATCH_LOAD_B32: partial(enc_vmem, 20, 1), RDNA3Ops.SCRATCH_LOAD_B64: partial(enc_vmem, 21, 1),
  RDNA3Ops.SCRATCH_LOAD_B96: partial(enc_vmem, 22, 1), RDNA3Ops.SCRATCH_LOAD_B128: partial(enc_vmem, 23, 1),
  RDNA3Ops.SCRATCH_STORE_B8: partial(enc_vmem, 24, 1), RDNA3Ops.SCRATCH_STORE_B16: partial(enc_vmem, 25, 1),
  RDNA3Ops.SCRATCH_STORE_B32: partial(enc_vmem, 26, 1), RDNA3Ops.SCRATCH_STORE_B64: partial(enc_vmem, 27, 1),
  RDNA3Ops.SCRATCH_STORE_B96: partial(enc_vmem, 28, 1), RDNA3Ops.SCRATCH_STORE_B128: partial(enc_vmem, 29, 1),
}

class RDNA3Renderer(ISARenderer):
  has_local = True
  extra_matcher = extra_matcher
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  post_isel_matcher = post_isel_matcher
  flag_remat_matcher = flag_remat_matcher
  post_regalloc_matcher = post_regalloc_matcher
  #post_regalloc_matcher2 = post_regalloc_matcher2
  code_for_op = {x: lambda: None for x in (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.NEG, Ops.SUB, Ops.CMPLT, Ops.CMPEQ)}

  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_amd import RDNA3Compiler
    self.compiler = RDNA3Compiler()
  def two_address(self, x:UOp) -> int:
    # masked loads are modelled as 2 address where the alt value is the reused src
    if isinstance(x.arg, RDNA3Ops) and x.arg.name.startswith(("SCRATCH_LOAD", "GLOBAL_LOAD", "DS_LOAD")) and len(x.src) == 5: return 3
    return 2 if x.arg in RDNA3GroupOp.TwoAddress else -1
  def supported_dtypes(self): return {d for d in super().supported_dtypes() if d not in dtypes.fp8s+dtypes.int8s+dtypes.int64s}
  # TODO: implement this
  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    for u in uops:
      if u.op is not Ops.INS: continue
      print(u.arg, u.tag, [reg_use(s) if s.op is not Ops.CONST else s.arg for s in u.src])
    return ""
  def render(self, uops:list[UOp]) -> str:
    #assert False
    targets: dict[str, int] = {}
    jumps: dict[UOp, int] = {}
    binary = bytearray()
    max_sgpr, max_vgpr = 0, 0
    n_bufs, n_vars, lds_size, scratch_size, gids = 0, 0, 0, 0, set()

    # encode the instructions and record kernel descriptors
    for u in uops:
      #if u.arg == RDNA3Ops.S_CMP_LT_I32 and u.src[1].arg == 5: assert False
      if u.op is Ops.PARAM: n_bufs += 1
      elif u.op is Ops.DEFINE_VAR: n_vars += 1
      elif u.op is Ops.DEFINE_LOCAL: lds_size += u.ptrdtype.size * u.ptrdtype.base.itemsize
      elif u.op is Ops.DEFINE_PRIVATE: scratch_size += u.ptrdtype.size * u.ptrdtype.base.itemsize
      elif u.op is Ops.SPECIAL and u.arg.startswith("gidx"): gids.add(int(u.arg[-1]))
      if u.op is not Ops.INS: continue
      if isinstance(reg:=u.reg, Register):
        if is_vgpr(reg): max_vgpr = max(max_vgpr, reg.index + reg.width // 32)
        elif is_sgpr(reg): max_sgpr = max(max_sgpr, reg.index + reg.width // 32)
      if u.arg is RDNA3Ops.LABEL:
        targets[u.tag] = len(binary)
        continue
      if u.arg not in encodings or (l:=encodings[u.arg](u)) is None:
        raise RuntimeError(f"failed to encode {u.arg} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      binary.extend(l)
      if u.arg in (RDNA3Ops.S_BRANCH, RDNA3Ops.S_CBRANCH_SCC0): jumps[u] = len(binary)
    # fixup jump targets now that encoding size is known
    for u in uops:
      if (t:=jumps.get(u)) is not None: binary[t-4:t-2] = ((targets[u.tag] - t) // 4).to_bytes(2, 'little', signed=True)

    # build the elf code object
    # ** pad text to ISA alignment
    padding_inst = int(0).to_bytes(4) #s_code_end().to_bytes()
    text = binary + padding_inst * ((hsa.AMD_ISA_ALIGN_BYTES - len(binary) % hsa.AMD_ISA_ALIGN_BYTES) % hsa.AMD_ISA_ALIGN_BYTES)
    text_offset = round_up(ctypes.sizeof(libc.Elf64_Ehdr), hsa.AMD_ISA_ALIGN_BYTES)
    # ** pack kernel descriptor (rodata)
    next_free_vgpr = round_up(max_vgpr, 8)
    vgpr_granule = max(0, (next_free_vgpr + 7) // 8 - 1)
    sgpr_granule = 0
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t()
    desc.group_segment_fixed_size = lds_size
    desc.private_segment_fixed_size = scratch_size
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
    #ehdr.e_ident[:6], ehdr.e_shoff, ehdr.e_shnum, ehdr.e_shstrndx = b"\x7FELF\x02\x01", shdr_offset, len(sections), 2

    ehdr.e_ident[:7] = b"\x7FELF\x02\x01\x01"      # +EI_VERSION = EV_CURRENT
    ehdr.e_ident[libc.EI_OSABI] = 64                # ELFOSABI_AMDGPU_HSA
    ehdr.e_ident[libc.EI_ABIVERSION] = 2            # AMDGPU code object v... (match your loader)
    ehdr.e_type      = libc.ET_REL                  # object file (1)
    ehdr.e_machine   = 224                          # EM_AMDGPU
    ehdr.e_version   = 1                            # EV_CURRENT
    ehdr.e_flags     = 0x041                         # EF_AMDGPU_MACH_AMDGCN_GFX1100 — pick per target
    ehdr.e_ehsize    = ctypes.sizeof(libc.Elf64_Ehdr)
    ehdr.e_shentsize = ctypes.sizeof(libc.Elf64_Shdr)
    ehdr.e_shoff, ehdr.e_shnum, ehdr.e_shstrndx = shdr_offset, len(sections), 2

    elf = bytearray(shdr_offset + ctypes.sizeof(shdrs))
    elf[0:ctypes.sizeof(ehdr)] = bytes(ehdr)
    elf[text_offset:text_offset+text_size] = text
    elf[rodata_offset:rodata_offset+rodata_size] = rodata
    elf[strtab_offset:strtab_offset+strtab_size] = strtab
    elf[shdr_offset:shdr_offset+ctypes.sizeof(shdrs)] = bytes(shdrs)
    binary = bytes(elf)

    return binary.hex()
