from tinygrad.device import Compiled, MallocAllocator
from tinygrad.renderer.arm64 import ARM64Renderer
from tinygrad.runtime.ops_clang import ClangCompiler, ClangProgram

class ARM64Device(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, ARM64Renderer(), ClangCompiler(cachekey="compile_arm64", args=['--target=aarch64-linux-gnu'], lang=['assembler']), ClangProgram)