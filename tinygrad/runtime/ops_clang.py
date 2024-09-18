from typing import Optional, List
import ctypes, subprocess, pathlib, tempfile, mmap
from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.helpers import cpu_time_execution, DEBUG, cpu_objdump
from tinygrad.renderer.cstyle import ClangRenderer

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:Optional[List[str]]=None):
    self.args = ['-march=native'] if args is None else args
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(['clang', *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-ffreestanding', '-c',
                               '-', '-o', output_file.name], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name: str, lib: bytes):
    if DEBUG >= 6: cpu_objdump(lib)

    self.name, self.lib = name, lib
    self.mapped_code = mmap.mmap(-1, len(lib), flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    self.mapped_code.write(lib[64:]) # offset is 64
    self.fxn = ctypes.CFUNCTYPE(None)(ctypes.addressof(ctypes.c_void_p.from_buffer(self.mapped_code)))

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)
    
  def __del__(self): self.mapped_code.close()

class ClangDevice(Compiled):
  def __init__(self, device:str):
    from tinygrad.runtime.graph.clang import ClangGraph
    super().__init__(device, MallocAllocator, ClangRenderer(), ClangCompiler(), ClangProgram, ClangGraph)
