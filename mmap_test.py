from tinygrad.runtime.ops_clang import ClangProgram, ClangCompiler, MallocAllocator

# allocate some buffers
out = MallocAllocator.alloc(4)
a = MallocAllocator.alloc(4)
b = MallocAllocator.alloc(4)

# load in some values (little endian)
MallocAllocator.copyin(a, bytearray([2,0,0,0]))
MallocAllocator.copyin(b, bytearray([3,0,0,0]))

# compile a program to a binary
lib = ClangCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")

# create a runtime for the program
fxn = ClangProgram("add", lib)

# run the program
fxn(out, a, b)

# check the data out
print(val := MallocAllocator.as_buffer(out).cast("I").tolist()[0])
