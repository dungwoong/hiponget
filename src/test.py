import os
import torch # if you add this import, you don't need to add dll directory!!
# os.add_dll_directory("C:/Users/kevin/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/lib")
import build.libv1 as libv1

a = torch.ones((16, 16), dtype=torch.float32)
addr = libv1.to_hip(a)
print(f'{addr=}')
libv1.double(addr, a.numel())
libv1.to_torch(addr, a)
print(a)