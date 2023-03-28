---
layout: post
title:  "External CUDA Allocator with PyTorch"
date:   2023-02-26 14:14:49 +0530
categories: python, pytorch
---

### External CUDA Allocator with PyTorch

#### Why does PyTorch need to support an external allocator?

Usually, you don't have to worry too much about how PyTorch allocates memory on CUDA (GPU) device when you train a model on GPU. But the problem arises when you have another library orchestrating some compute on GPU with its own allocator. A concrete example of this would be, lets say, we are using cuDF (think pandas for GPU) to do some processing on a data-frame and then convert that dataframe to train your model with PyTorch. In this case, cuDF will have its own allocator which will allocate some memory for the dataframe and post the processing when we create Tensors from that dataframe, PyTorch will allocate using its allocator. So what happens now is cuDF allocator will mark the memory used for dataframe as free but it will still keep that memory with itself for future (in case there is request for memory). This means when PyTorch tries to allocate it sees less free memory (even though cuDF is not using that memory) and this can lead to the dreaded Out-of-Memory errors (OOM).

NOTE: The reason cuDF's memory allocator keeps memory is because asking the system for memory is not cheap. So allocators are future-looking and avoid constant allocations and deallocations of their memory-pools.

#### What can be done?

By now, we know that memory on GPU is a relatively precious resource (as compared to CPU memory) and we are convinced that using two libraries with their own allocator to compute something on GPU could lead to OOM errors. So, we understand that the fundamental problem is that two allocators are competing for a resource and both don't have any idea about the existence of other. We can solve this problem potentially in two ways.

1. Both libraries share the same allocator. This way, this single allocator has the complete picture and control of the GPU memory usage and can allocate the freed dataframe from our example to the Tensors.
2. Allocators can talk to each-other and query some details about the memory that they are holding. This way both libraries can have their own allocators optimal for their use-cases but they share some information (and maybe memory) with others libaries.

#### Support for External Allocator in PyTorch

This [issue](https://github.com/pytorch/pytorch/issues/43144) regarding the support for external allocator has all the context on supporting the external allocator in PyTorch. We will summarize some important points from the issue

* Using PyTorch with cuDF can lead to OOM even though this could be avoided if both PyTorch and cuDF shared the allocator.
* One problem with allowing external allocator is that if some library opaquely replaces the allocator with a sub-optimal one, then user will get the impression that PyTorch is slow (even though the allocator could be slowing it down).
* Counter argument for above point is that even with OOM from PyTorch, user still gets the impression that PyTorch is the problem.
* It also discusses what the API for allocator should be and if there are any standard for this API.

[Pull Request](https://github.com/pytorch/pytorch/pull/86786) by [emcastillo](https://github.com/emcastillo) from Preferred Networks took care of fixing this issue by adding the necessary infrastructure to PyTorch. With this pull request user can now easily swap the CUDA allocator for PyTorch. However, one thing to note is that you can't swap an already initialized allocator (as it may have already allocated some memory).

#### API

As [documented](https://pytorch.org/docs/master/notes/cuda.html#using-custom-memory-allocators-for-cuda), we can see that our allocator should be a shared library which implements two functions for allocation and de-allocation. Signature for the allocator function should be `void* alloc(ssize_t size, int device, cudaStream_t stream)` and for deallocator should be `void free(void* ptr, ssize_t size, int device, cudaStream_t stream)`. Note that the names of the function don't matter as we will inform PyTorch about it.

Example
```c++
// filename alloc.cc
#include <cuda_runtime_api.h>
#include <iostream>

extern "C" {

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);
   std::cout<<"alloc "<<ptr<<size<<std::endl;
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
   cudaFree(ptr);
}

}
```

In the above example (from docs), we don't do anything fancy but just forward the `malloc` and `free` calls to `cudaMalloc` and `cudaFree` with some prints. We will compile the above code with `g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC`.

Now, we can use this `alloc.so` with PyTorch
```python
import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    'alloc.so', 'my_malloc', 'my_free')

# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)

# This will allocate memory in the device using the new allocator
b = torch.zeros(10, device='cuda')
```

#### The Big Picture : RMM

Now, it is nice that one can ask PyTorch to accept external allocator but it would be tedious if we had to write our own allocators. It is certainly not a trivial task. To this end, Rapidsai already has RMM (Rapids Memory Manager) which can be used as the allocator for `cuDF`, `cuPy`, `numba`, etc. and with the above PR in place, also with `PyTorch`. So, if you are using `cuDF` with `PyTorch` you can easily configure your script such that both use RMM which will lead to better management of device memory.

Using PyTorch with RMM
```python
import rmm
import torch

mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.ManagedMemoryResource())
rmm.mr.set_current_device_resource(mr)
torch.cuda.change_current_allocator(rmm.rmm_torch_allocator)

x = torch.randn(100).cuda()

# the memory resource reports information about PyTorch allocations
print(mr.allocation_counts)
# PyTorch by default uses `float32`, so 100 `float32` will take 400 bytes.
# {'current_bytes': 400, 'current_count': 1, 'peak_bytes': 400, 'peak_count': 1, 'total_bytes': 400, 'total_count': 1}
```

Using PyTorch and cuPy with RMM
```python
import rmm
import torch
import cupy
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

mr = rmm.mr.StatisticsResourceAdaptor(rmm.mr.ManagedMemoryResource())
rmm.mr.set_current_device_resource(mr)
torch.cuda.change_current_allocator(rmm.rmm_torch_allocator)

c = cupy.random.randn(100)
t = torch.randn(100).cuda()

# the memory resource reports information about allocations
print(mr.allocation_counts)
# cuPy by default uses `float64`, so 100 `float64` will take 800 bytes.
# {'current_bytes': 1200, 'current_count': 2, 'peak_bytes': 1200, 'peak_count': 2, 'total_bytes': 1200, 'total_count': 2}
```

#### Conclusion

* PyTorch now supports external allocators which confirm to the interface.
* RMM provides an implementation compatible with PyTorch.
* Users can now use multiple libraries which compute on GPU with tighters control on GPU memory.

References:
* [Issue requesting the feature](https://github.com/pytorch/pytorch/issues/43144)
* [PyTorch PR adding the infrastructure](https://github.com/pytorch/pytorch/pull/86786)
* [RMM PR adding support for integrating with PyTorch](https://github.com/rapidsai/rmm/pull/1168)
