---
layout: post
title:  "External CUDA Allocator with PyTorch"
date:   2023-02-26 14:14:49 +0530
categories: python, pytorch
---

### External CUDA Allocator with PyTorch

#### Why does PyTorch need to support an external allocator?

Usually, you don't have to worry to much about how PyTorch allocates memory on CUDA (GPU) device when you train a model on GPU. But the problem arises when you have another library orchestrating some compute on GPU with its own allocator. A concrete example of this would be, lets say, we are using cuDF (think pandas for GPU) to do some processing on a data-frame and then convert that dataframe to train your model with PyTorch. In this case, cuDF will have its own allocator which will allocate some memory for the dataframe and post the processing when we create Tensors from that dataframe, PyTorch will allocate using its allocator. So what happens now is cuDF allocator will mark the memory used for dataframe as free but it will still keep that memory with itself for future (in case there is request for memory). This means when PyTorch tries to allocate it sees less free memory (even though cuDF is not using that memory) and this can lead to the dreaded Out-of-Memory errors (OOM).

NOTE: The reason cuDF's memory allocator keeps memory is because asking the system for memory is not cheap. So allocators are future-looking and avoid constant allocations and deallocations of their memory-pools.

#### What can be done?

By now, we know that memory on GPU is a relatively precious resource (as compared to CPU memory) and we are convinced that using two libraries with their own allocator to compute something on GPU could lead to OOM errors. So, we understand that the fundamental problem is that two allocators are competing for a resource and both don't have any idea about the existence of other. So, we can solve this problem potentially in two ways.

1. Both libraries share the same allocator. This way, this single allocator has the complete picture and control of the GPU memory usage and can allocate the freed dataframe from our example to the Tensors.
2. Allocators can talk to each-other and query some details about the memory that they are holding. This way both libraries can have their own allocators optimal for their use-cases but they share some information (and maybe memory) with others libaries. 

Ref: 
https://github.com/pytorch/pytorch/issues/43144
https://github.com/rapidsai/rmm/pull/1168