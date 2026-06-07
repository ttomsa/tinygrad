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
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  @property
  def cons(self): return self._cons or (self,)
  def __repr__(self): return self.name

@dataclass(frozen=True)
class TupleRegister:
  regs: tuple[Register, ...]
  @property
  def index(self): return self.regs[0].index
  #def __repr__(self): return f"{self.regs[0]}:"
  def __getitem__(self, key): return self.regs[key]

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(sink.toposort())
    self.reg_n = itertools.count()
    arg_order = {Ops.PARAM: 0, Ops.DEFINE_VAR: 1, Ops.SPECIAL: 2}
    self.func_args = sorted([u for u in self.uses if u.op in arg_order], key=lambda k: (arg_order[k.op], k.arg))

  def is_foldable(self, x:UOp, s:UOp) -> bool: return len(self.uses[s]) == x.src.count(s) == 1
  def vreg(self, cons:tuple[Register, ...]|Register):
    return Register(f"v{next(self.reg_n)}", 0, _cons=cons if isinstance(cons, tuple) else (cons,))

def imm(dt:DType, v:int) -> UOp: return UOp.const(dt, truncate[dt](v)).rtag()
def def_reg(dt:DType, reg:Register|None=None) -> UOp: return UOp(Ops.DEFINE_REG, dt, tag=None if reg is None else (reg,))

@dataclass
class PreRegAllocContext:
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)

class ISARenderer(Renderer):
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher|None = None
  post_regalloc_matcher: PatternMatcher

  def is_two_address(self, x:UOp) -> bool: return False
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, disp:UOp, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, disp:UOp, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")