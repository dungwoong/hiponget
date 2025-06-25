import os
import time
import torch # if you add this import, you don't need to add dll directory!!
# os.add_dll_directory("C:/Users/kevin/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/lib")
import build.libv1 as libv1

m, n, k = 1024, 1024, 1024
a = torch.randn((m, k), dtype=torch.float32)
b = torch.randn((k, n), dtype=torch.float32)
c = torch.randn((m, n), dtype=torch.float32)
a_addr = libv1.to_hip(a)
b_addr = libv1.to_hip(b)
c_addr = libv1.to_hip(c)
libv1.matmul(a_addr, b_addr, c_addr, m, n, k)
libv1.to_torch(c_addr, c)

expected = a @ b
print(c)
print(expected)
assert torch.allclose(c, expected, atol=0.002), torch.max(c - expected)
print(f"Verified :)\nMax diff {torch.max(c - expected)}")

iters = 10
flops = iters * 2 * m * n * k

print("Starting HIP kernel profiling...")
start = time.time()
for i in range(iters):
    libv1.matmul(a_addr, b_addr, c_addr, m, n, k)
    libv1.to_torch(c_addr, c)
finish = time.time()

print(f'avg time {(finish - start) / iters}')
print(f'gflops/s {(flops / (finish - start)) * 1e-9}')
print('\n')

print("Starting torch profiling...")
start = time.time()
for i in range(iters):
    c = a @ b
finish = time.time()

print(f'avg time {(finish - start) / iters}')
print(f'gflops/s {(flops / (finish - start)) * 1e-9}')
print('\n')

print("Starting numpy profiling...")
a_numpy = a.numpy()
b_numpy = b.numpy()
start = time.time()
for i in range(iters):
    c_numpy = a_numpy @ b_numpy
finish = time.time()

print(f'avg time {(finish - start) / iters}')
print(f'gflops/s {(flops / (finish - start)) * 1e-9}')
print('\n')