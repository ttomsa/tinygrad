from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, consumer_map_from_toposort
from tinygrad.dtype import DType, truncate

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  width: int = 0
  # the register units that comprise self, a unit is the smallest addressable register, this only applies to reals
  _units: tuple[Register, ...] = field(default_factory=tuple)
  # the legal registers that can be assigned to self, this only applies to virtuals
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  @property
  def units(self): return self._units or (self,)
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name
  def subreg(self, i:int, width:int) -> Register:
    urng = width // self.units[0].width
    uidx = i * urng
    return Register(self.name, self.units[uidx].index, width, self.units[uidx:uidx+urng])

# this is called to retrieve the register x outputs (outputs != defines)
# when x is GETTUPLE there are three cases according to what x's src defines:
#  1: it defines multiple regs, retrieve reg at position x.arg
#  2: it defines a single reg with sub regs, retrieve the sub reg at position x.arg
#  3: it defines a single reg with no sub regs, retrieve that reg
# case 2 happens when printing assembly or encoding because real regs may have sub regs
# case 3 happens in regalloc because virtuals don't have sub virtuals and are treated as a single allocatable value
def reg_use(x:UOp, i:int|None=None) -> Register|None:
  if isinstance(x.tag, tuple):
    if len(x.tag) > 1 and i is not None:
      assert isinstance(reg:=x.tag[i], Register)
      return reg
    assert isinstance(reg:=x.tag[0], Register)
    return reg.subreg(i, x.dtype.scalar().bitsize) if i is not None and reg._units else reg
  if x.op is Ops.GETTUPLE: return reg_use(x.src[0], x.arg)
  if x.op in (Ops.NOOP, Ops.AFTER) and x.src: return reg_use(x.src[0])
  return None

def reg_uses(x:UOp) -> tuple[Register, ...]: return tuple(r for s in x.src if (r:=reg_use(s)) is not None)
def reg_defs(x:UOp) -> tuple[Register, ...]:
  if isinstance(x.tag, tuple):
    assert all(isinstance(r, Register) for r in x.tag)
    return x.tag
  return ()

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_args = sorted([u for u in self.uses if u.op in arg_order], key=lambda k: (arg_order[k.op], k.arg))

  def is_foldable(self, x:UOp, s:UOp) -> bool: return len(self.uses[s]) == x.src.count(s) == 1
  def vreg(self, cons:tuple[Register, ...]|Register):
    return Register(f"v{next(self.reg_n)}", self.reg_n, _cons=cons if isinstance(cons, tuple) else (cons,))

def imm(dt:DType, v:int) -> UOp: return UOp.const(dt, truncate[dt](v)).rtag()
def def_reg(dt:DType, reg:Register|None=None) -> UOp: return UOp(Ops.DEFINE_REG, dt, tag=None if reg is None else (reg,))

@dataclass
class FlagRematContext:
  flags: dict[Register, UOp] = field(default_factory=dict)

@dataclass
class PostRegallocContext:
  issued: dict[int, int] = field(default_factory=dict)
  pending: dict[Register, tuple[int, int]] = field(default_factory=dict)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  post_isel_matcher: PatternMatcher|None = None
  flag_remat_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher
  post_regalloc_matcher2: PatternMatcher|None = None

  def two_address(self, x:UOp) -> int: return -1
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")