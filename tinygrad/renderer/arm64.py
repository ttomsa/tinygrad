from typing import List, Dict, cast
from tinygrad.ops import UOp, Ops, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad import dtypes
from tinygrad.dtype import DType, PtrDType
import struct

arm64_mov_ops = {Ops.STORE: "str", Ops.LOAD: "ldr", Ops.ASSIGN: "str", Ops.DEFINE_ACC: "ldr"}
arm64_unsigned_ops = {**arm64_mov_ops, Ops.ADD: "add", Ops.SUB: "sub", Ops.MUL: "mul", Ops.IDIV: "udiv", Ops.MOD: "udiv", Ops.CMPNE: "cmp",
                      Ops.CMPLT: "cmn", Ops.AND: "and", Ops.OR: "orr", Ops.XOR: "eor", Ops.SHL: "lsl", Ops.SHR: "lsr"}
arm64_signed_ops = {**arm64_unsigned_ops, Ops.IDIV: "sdiv", Ops.MOD: "sdiv", Ops.SHR: "asr"}
arm64_float_ops = {Ops.ADD: "fadd", Ops.SUB: "fsub", Ops.MUL: "fmul", Ops.FDIV: "fdiv", Ops.CMPLT: "fcmp", Ops.CMPNE: "fcmp",
                     Ops.SQRT: "fsqrt", **{k:v for k,v in arm64_mov_ops.items()}}
arm64op = {**{x:arm64_unsigned_ops for x in (dtypes.bool,)+dtypes.uints}, **{x:arm64_signed_ops for x in dtypes.sints},
         **{x:arm64_float_ops for x in dtypes.floats}}

arm64_i_reg_map = {**{f"x{i}": {4: f"w{i}"} for i in range(0,29)}}
arm64_f_reg_map = {**{f"v{i}": {8: f"d{i}", 4: f"s{i}", 2: f"h{i}", 1: f"b{i}"} for i in range(0,32)}}

arm64_rewrite = PatternMatcher([
  (UPat(Ops.INDEX, src=(UPat(), UPat(Ops.CONST)), name="x"), lambda ctx,x: f"add {ctx[x]}, {ctx[x.src[0]]}, {x.src[1].arg*x.src[0].dtype.itemsize}"),
  (UPat(Ops.INDEX, name="x"), lambda ctx,x: f"add {ctx[x]}, {ctx[x.src[0]]}, {ctx[x.src[1]]}, lsl #{x.src[0].dtype.itemsize.bit_length()-1}"),
  #(UPat(Ops.LOAD, name="x"), lambda ctx,x: f""),
  (UPat(Ops.LOAD, name="x"), lambda ctx,x: f"{arm64op[x.dtype][x.op]} {ctx[x]}, [{ctx[x.src[0]]}]"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"{arm64op[x.src[1].dtype][x.op]} {ctx[x.src[1]]}, [{ctx[x.src[0]]}]"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x: f"{arm64op[x.dtype][x.op]} {ctx[x]}, {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
])

arm64_matcher = PatternMatcher([
  # *** also in ptx ***
  # cast between pointers is a noop
  (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) else None),
  # *** also in llvmir ***
  # rewrite cast to bool to CMPNE 0
  (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
  # rewrite RECIP to FDIV
  (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
  # rewrite MAX to CMPLT + WHERE
  (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
])

class ARM64Renderer(Renderer):
  device = "ARM64"
  supports_float4 = False
  has_local = False
  has_shared = False
  global_max = None
  extra_matcher = arm64_matcher

  def regt(self, reg:str, dt:DType) -> str:
    if dt.itemsize == 8 or isinstance(dt, PtrDType): return reg
    if dtypes.is_float(dt): arm64_f_reg_map[reg][dt.itemsize]
    return arm64_i_reg_map[reg][dt.itemsize]

  def __getitem__(self, key:UOp): return self.regt(self.r[key], key.dtype) if self.r[key] in self.all_regs else self.r[key]  # hacky helper
  def render(self, name:str, uops:List[UOp]) -> str:
    # 64 bit general registers x29-31 not included, x28 temp register
    gen_regs = [f"x{i}" for i in range(0,28)]
    float_regs = [f"v{i}" for i in range(0,32)]
    self.all_regs = gen_regs + float_regs
    # can be a register, memory location or immediate value
    r: Dict[UOp, str] = {}
    self.r = r
    mem: Dict[UOp, str] = {}
    stack_size: int = 8
    arg_stack_offset: int = 16
    kernel: List[str] = []
    self.uops = uops
    last_use: Dict[UOp, int] = {var: i for i,u in enumerate(uops) for var in (v for v in (u,) + u.src if v.dtype != dtypes.void)}

    def is_imm(u:UOp) -> bool: return u.op is Ops.CONST and not dtypes.is_float(u.dtype) and abs(u.arg) <= dtypes.max(dtypes.int32)
    def is_mem(u:UOp) -> bool: return u in r and u in mem and r[u] == mem[u]
    def is_reg(loc:str) -> bool: return loc in self.all_regs
    def free_reg(reg:str): float_regs.insert(0, reg) if reg.startswith("xmm") else gen_regs.insert(0, reg)

    def mov_to_reg(u:UOp, reg:str):
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or reg == "r15" else u.dtype
      kernel.append(f"{arm64op[dt][Ops.LOAD]} {reg}, {r[u]}")
      r[u] = reg

    def mov_to_stack(u:UOp):
      nonlocal stack_size
      if u not in mem:
        mem[u] = f"[x29, #-{stack_size}]"
        stack_size += 8
      dt = dtypes.int64 if isinstance(u.dtype, PtrDType) or r[u] == "r15" else u.dtype
      kernel.append(f"{arm64op[dt][Ops.STORE]} {mem[u]}, {r[u]}")
      r[u] = mem[u]

    def assign_reg(i:int, dt:DType) -> str:
      type_regs = float_regs if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else gen_regs
      if type_regs: return type_regs.pop(0)
      t = 'x' if dtypes.is_float(dt) and not isinstance(dt, PtrDType) else 'r'
      # TODO: remove range check
      candidates = [u for u in r if r[u][0] == t and u not in (uops[i],) + uops[i].src and u.op is not Ops.RANGE]
      chosen = max(candidates, key=lambda u: last_use[u])
      reg = r[chosen]
      mov_to_stack(chosen)
      return reg

    for i,u in enumerate(uops):
      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = assign_reg(i, u.dtype)
        if r[u] in ("x8", "v8"): # value is in stack instead of register, rbp + 8 is return address
          free_reg(r[u])
          r[u] = mem[u] = f"[x29, #{arg_stack_offset}]"
          arg_stack_offset += 8
      elif u.op is Ops.CONST: r[u] = f"#{u.arg}"
      else:
        for s in u.src: # mov srcs
          # these can't take imm values
          if is_imm(s) and not is_reg(r[s]) and u.op in (Ops.WHERE, Ops.IDIV, Ops.MOD): mov_to_reg(s, assign_reg(i, s.dtype))
          elif is_mem(s): mov_to_reg(s, assign_reg(i, s.dtype))
        if u.dtype != dtypes.void: # assign destination
          if u.op is Ops.ASSIGN: r[u] = mem[u] = mem[u.src[0]] # define acc was already spilled here
          else: r[u] = assign_reg(i, u.dtype)
        if u.op is Ops.RANGE: # all registers get moved to stack before loop TODO: remove range check
          for var in (v for v in r if is_reg(r[v]) and v.op is not Ops.RANGE):
            free_reg(r[var])
            mov_to_stack(var)
          last_use[u.src[1]] = max(last_use[u], last_use[u.src[1]])
        # render x86 assembly
        if (l:=arm64_rewrite.rewrite(u, ctx=self)) is None:
          raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))
      # free dead registers
      for loc in (r.pop(v) for v in (u,) + u.src if v in r and last_use[v] == i):
        if is_reg(loc) and loc not in r.values(): free_reg(loc)

    return "\n".join([".text", f".global {name}", f"{name}:", "stp x29, x30, [sp, #-16]!", "mov x29, sp", f"sub sp, sp, #{stack_size}"] + \
                      kernel + [f"add sp, sp, #{stack_size}", "ldp x29, x30, [sp], #16", "ret", "\n"])