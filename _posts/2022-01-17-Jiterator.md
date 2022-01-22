---
layout: post
title:  "JITerator"
date:   2022-01-17 14:14:49 +0530
categories: pytorch, cuda
---

PyTorch is one of the leading Deep Learning frameworks. It supports a lot of operations to operate on Tensor.
This rich set of operators allows researchers to prototype new solutions to their problems with shorter iterations. As these operators
are the basic building block of any PyTorch script, they must be as performant as possible. Writing performant
operators with parallelization while keeping the cache hot is not easy. Writing these operators given that `Tensor`s are not always laid out contiguously in memory, is tricky. Handling these myriads of cases while maintaining performance is not a trivial development task.

## Enter the TensorIterator!
But we have TensorIterator to the rescue! PyTorch internals has a developer-friendly infrastructure to do the same. It helps to write performant code while handling
all these edge cases. So all that the developer needs to add is the actual computation for the operator. When using TensorIterator,
the developer only has to specify the `inputs` and `outputs` for the operator and transformations from inputs to the output value.
TensorIterator also has extra checks and features to make sure `inputs` and `outputs` do not overlap, they are on the same `device`, have the same `dtype`, etc. It also takes care of type promotion on setting the correct flags.

Example code using `TensorIterator` (from first blog reference)
```c++
at::TensorIteratorConfig iter_config;
iter_config
  .add_output(c)
  .add_input(a)
  .add_input(b);

auto iter = iter_config.build();

// Element-wise add
at::native::cpu_kernel(iter, [] (float a, float b) -> float {
  return a + b;
});
```

A ton of PyTorch operators use TensorIterator internally. TensorIterator suits very well for elementwise operations like Unary Operators (`exp`, `sin`, `log`, etc.) and Binary Operators (`add`, `mul`, etc.). It also works well for Reduction Operators like (`sum`, `mean`, `prod`, etc.).

For more details on `TensorIterator`, please refer to the following blogs that talk about it's internals and show example usage.

Blogs:
  * [Quansight TensorIterator Update](https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update/index.html)
  * [Quansight TensorIterator](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals/)

## TensorIterator and CUDA

The existing machinery for using TensorIterator to generate CUDA Kernels of the operation is easy to use. But there is one major caveat in the process. The helper functions like `cpu_kernel` and `gpu_kernel` that generate the kernel code for the operator are heavily templated. These templates are instantiated for all of the opeartor's supported dtypes. They also generate different kernels based on whether the Tensors are contiguous or not. If a binary operator supports scalar argument, then two more kernels are instantiated per dtype for `Tensor-Scalar` and `Scalar-Tensor` case. All of this leads to an increase in build time especially for `nvcc` (CUDA AOT compiler). Also, these kernels are compiled using CUDA Runtime API, which means that these kernels are loaded when PyTorch binary is loaded thus increasing the CUDA context size (consuming actual GPU VRAM). These issues became very apparent while adding new operators for `torch.special` module. A user who didn't care about these operators had to pay the cost (in terms of GPU memory) when importing PyTorch with CUDA support.

## The JITerator

The solution to these problems that the PyTorch maintainers' Mike Ruberry and Natalia Gimelshein came up with was to use NVRTC (Runtime Compilation) library shipped with CUDA to delay the compilation and loading of these kernels. Another way to put it,
we will not compile the kernel and load the corresponding kernel binary till the operator is first called. With this approach, there is no upfront compilation cost. The operator kernel code will be loaded in the GPU memory only if the operator is ever used.

The way this works is when PyTorch is built, it keeps the string representation of the kernel code. This string is utilized to compile the kernel when a jitted operator is called. So when a user calls the operator, we check in the cache if there is already a compiled kernel available, if not NVRTC is utilized to generate the kernel and load it for use. The address of this generated kernel is cached so that next time this operator is called, we can reuse this kernel. We will expand this idea in the following section and dive deeper into JITerator.

* How to use JITerator with TensorIterator
    * [Computation String](#computation-string) (`jiterator_stringify`)
    * [Generating the Kernel](#generating-the-kernel)
* [Diving Deeper](#diving-deeper)
    * [`jitted_gpu_kernel`](#`jitted_gpu_kernel`)
    * `jitted_gpu_kernel_impl`
    * NVRTC JIT utility helpers
    * `launch_jitted_unrolled_kernel` and `launch_jitted_vectorized_kernel`

### Computation String
Let us take a look at how the said computation string looks in code. We will look at the implementation of the binary operator `gcd`. Below is the code for computing `gcd` of two numbers `a_in` and `b_in`.
Do note that the code does not look like a string. That is because of the macro `jiterator_stringify` that converts the code to string. Thanks to this we do not lose out on syntax highlighting and the code still feels like code even if we are getting the string of the code in the variable `gcd_string`.
```c++
const auto gcd_string = jiterator_stringify(
  template <typename T>
  T gcd(const T a_in, const T b_in) {
    T a = abs(a_in);
    T b = abs(b_in);

    while (a != T{0}) {
      T c = a;
      a = b % a;
      b = c;
    }

    return b;
  }
); // gcd_string
```

### Generating the Kernel
Now that we have the string representation of our computation, we need to set up the TensorIterator and pass it to the JITerator machinery that will hold onto this computation string and compile it once the jitted operator is first called.

```c++
// Setting up the TensorIterator
at::TensorIteratorConfig iter_config;
iter_config
  .add_output(result)
  .add_input(a)
  .add_input(b);

auto iter = iter_config.build();
...

// Defining the kernel.
const char gcd_name[] = "gcd";
void gcd_kernel_cuda(TensorIteratorBase& iter) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cuda", [&]() {
      jitted_gpu_kernel</*name=*/gcd_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 2>(iter, gcd_string);
    });
}
```

`jitted_gpu_kernel` is the entry point for the JITerator. It takes a name for the kernel, return dtype, computation dtype and the number of input arguments as template parameter. It takes TensorIterator `iter` and the computation `gcd_string` as run-time arguments. At this point, we are done with implementing the CUDA operator kernel. When this operator is called, `jitted_gpu_kernel` will take care of compiling the kernel and loading it for use.

## Diving Deeper

### `jitted_gpu_kernel`

Permanent Link : [Link](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/Loops.cuh#L111-L113)

As mentioned above, this is the entry point for JITerator. 

Following is a simplified C++ pseudo-code

```c++
// Entrypoint for jitted GPU kernels.
// Only handles elementwise unary and binary kernels with a
//   common dtype and a single output.
// NOTE: this assumes the op's iterator has a common_dtype.
template <char const *name, typename return_type, typename f_inputs_type, int arity>
void jitted_gpu_kernel(TensorIteratorBase& iter, const std::string& f,
                       at::cuda::jit::BinaryFuncVariant scalar_pos=at::cuda::jit::BinaryFuncVariant::NoScalar,
at::opmath_type<f_inputs_type> scalar_val=0) {
  // Checks
  ...

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel<name, return_type, f_inputs_type, arity>(sub_iter, f, scalar_pos, scalar_val);
    }
    return;
  }

  // Computes if dynamic casting is needed
  bool needs_dynamic_casting = false;

  // based on input and output dtype determine
  // if dynamic casting is needed
  ...

  // call `jitted_gpu_kernel_impl`
  if (scalar_pos == at::cuda::jit::BinaryFuncVariant::NoScalar) {
    jitted_gpu_kernel_impl</*name*/ name,
                    /*return_type=*/ return_type,
                    /*f_inputs_type=*/ f_inputs_type,
                    arity, at::cuda::jit::BinaryFuncVariant::NoScalar>(iter, f, needs_dynamic_casting);
  } else if (scalar_pos == at::cuda::jit::BinaryFuncVariant::RhsScalar) {
    jitted_gpu_kernel_impl</*name*/ name,
                    /*return_type=*/ return_type,
                    /*f_inputs_type=*/ f_inputs_type,
                    arity, at::cuda::jit::BinaryFuncVariant::RhsScalar>(iter, f, needs_dynamic_casting, scalar_val);

  } else {
    jitted_gpu_kernel_impl</*name*/ name,
                    /*return_type=*/ return_type,
                    /*f_inputs_type=*/ f_inputs_type,
                    arity, at::cuda::jit::BinaryFuncVariant::LhsScalar>(iter, f, needs_dynamic_casting, scalar_val);

  }
}
```

#### Walkthrough

This function does a few checks on `input`, `output` and figures out if the computation requires dynamic casting. Also, if the input size is larger than what can be handled by 32-bit indexing, we divide the input and output into 32-bit indexable blocks and call this function recursively. Finally it calls `jitted_gpu_kernel_impl`.

### `jitted_gpu_kernel_impl`

Permanent Link : [Link](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L279-L281)

```c++
template <char const *name, typename result_type, typename compute_type, int arity,
          at::cuda::jit::BinaryFuncVariant scalar_pos=at::cuda::jit::BinaryFuncVariant::NoScalar>
void jitted_gpu_kernel_impl(TensorIteratorBase& iter, const std::string& f, const bool dynamic_casting, compute_type scalar_val = 0) {
  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  // get data pointers to input and output array.
  constexpr int ntensors = arity + 1;
  at::detail::Array<char*, ntensors> data;
  for (auto i = decltype(ntensors){0}; i < ntensors; ++i) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  // Decides which of 4 kernel types to launch
  // Variations are:
  //   - Case 1: no dynamic casting and contiguous
  //   - Case 2: no dynamic casting and noncontiguous
  //   - Case 3: dynamic casting and contiguous
  //   - Case 4: dynamic casting and noncontiguous
  // These cases align with the non-jitted CUDALoops.cuh cases in gpu_kernel_impl

  if (!dynamic_casting) {
    if (contiguous) {
      // Case 1: no dynamic casting and contiguous
      launch_jitted_vectorized_kernel<name, result_type, compute_type, arity, scalar_pos>(
        iter.device().index(), numel, f, data, scalar_val);
      return;
    }

    // Case 2: no dynamic casting and noncontiguous
    auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
    auto output_offset_calculator = make_output_offset_calculator(iter);
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
      iter.device().index(), numel, f, data, input_offset_calculator,
      output_offset_calculator, loader, storer, contiguous, scalar_val);
    return;
  }

  // Cases 3 and 4 are handled below
  // Both require construction of a storer (this asserts 1 output) and one or more loaders

  // Creates store cast to output (the zeroth tensor in TensorIterator)
  auto storer = memory::StoreWithCast(iter.dtype(0));

  // Creates load casts from inputs (note offset indexing into the iterators 1...n tensors)
  at::detail::Array<ScalarType, arity> dtypes;
  for (auto i = decltype(arity){0}; i < arity; ++i) {
    dtypes[i] = iter.dtype(i + 1);
  }
  auto loader = memory::LoadWithCast<arity>(dtypes);

  if (contiguous) {
    // Case 3: dynamic casting and contiguous
    auto input_offset_calculator = TrivialOffsetCalculator<arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
      iter.device().index(), numel, f, data, input_offset_calculator,
      output_offset_calculator, loader, storer, contiguous, scalar_val);
    return;
  }

  // Case 4: dynamic casting and noncontiguous
  auto input_offset_calculator = make_input_offset_calculator<arity>(iter);
  auto output_offset_calculator = make_output_offset_calculator(iter);
  launch_jitted_unrolled_kernel<name, result_type, compute_type, scalar_pos>(
    iter.device().index(), numel, f, data, input_offset_calculator,
    output_offset_calculator, loader, storer, contiguous, scalar_val);
}

```
#### Walkthrough

`jitted_gpu_kernel_impl` figures out if the tensors are contiguous or not and adds extra machinery if they are non-contiguous and if dynamic casting is required (computed in `jitted_gpu_kernel`). Post that it passes the data to either `launch_jitted_unrolled_kernel` and `launch_jitted_vectorized_kernel`

### NVRTC JIT utility helpers

Before we look at `launch_jitted_vectorized_kernel` or `launch_jitted_unrolled_kernel`, let us take a quick look at the JIT utility functions that they call.

These utility functions are declared in `aten/src/ATen/native/cuda/jit_utils.h` [file](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/jit_utils.h) and defined in `aten/src/ATen/native/cuda/jit_utils.cu` [file](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/jit_utils.cu)

Important utility helpers to know about
* `generate_code`: This function takes the computation string and wraps it with the required machinery of reading from input, passing that data to kernel and writing to the output. It returns a string. This the string which is passed to NVRTC.

* `jit_vectorized_code_template` and `jit_code_template` : These strings variables are used to wrap the actual computation string and to create the actual kernel string. They take care of efficiently loading the input, calling the computation code on them and storing output tensor.

* `jit_pwise_function`: This function takes as input the string generated from `generate_code` and kernel name. It uses `NVRTC` functions to compile the code, loads that compiled code into the GPU VRAM and returns a pointer to the kernel function.

* `launch_jitted_pwise_function` : This function takes the kernel pointer we got from `jit_pwise_function` with the arguments for the kernel and launches the kernel with CUDA Driver API.

### `launch_jitted_unrolled_kernel` and `launch_jitted_vectorized_kernel`

Permanent Link : [Link1](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L126-L135) and [Link2](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L179-L186)

Besides using the machinery of consuming vectorized code, these functions are identical. So we will only look at one of them.

We will look at `launch_jitted_vectorized_kernel`.
```C++
template<
  char const *name,
  typename result_type,
  typename f_inputs_type,
  int arity,
  at::cuda::jit::BinaryFuncVariant scalar_pos,
  typename array_t>
static inline void launch_jitted_vectorized_kernel(DeviceIndex dev_idx, int64_t N, const std::string& f, array_t data,
at::opmath_type<f_inputs_type> scalar_val) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  const int64_t grid = (N + block_work_size() - 1) / block_work_size();
  const int vec_size = memory::jitted_can_vectorize_up_to<result_type, f_inputs_type, arity>(data);

  // Different kernels are compiled depending on what we're vectorizing up to (1, 2 or 4 elements)
  //   fn_ptr is set to the appropriate function based on the vec size and GPU used
  // TODO: Memory use can probably be optimized by re-using kernels across GPUs with
  //   the same compute capability
  static std::mutex _jiterator_mutex;
  static std::vector<at::cuda::jit::NvrtcFunction> fns4(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns2(c10::cuda::device_count());
  static std::vector<at::cuda::jit::NvrtcFunction> fns1(c10::cuda::device_count());


  at::cuda::jit::NvrtcFunction* fn_ptr;
  if (vec_size == 4) {
    fn_ptr = &fns4[dev_idx];
  } else if (vec_size == 2) {
    fn_ptr = &fns2[dev_idx];
  } else if (vec_size ==1) {
    fn_ptr = &fns1[dev_idx];
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected vec_size for jitter vectorized kernel");
  }

  bool vectorized = vec_size > 1;

  if (!fn_ptr->function) {
    // generate code if fn_ptr->function is nullptr
    // i.e. we haven't compiled it previously.
    const std::lock_guard<std::mutex> lock{_jiterator_mutex};
    if (!fn_ptr->function) {
      constexpr int nTensors = array_t::size();
      std::string string_name{name};
      std::string f_inputs_type_str = at::cuda::jit::typeName<f_inputs_type>();
      std::string compute_type_str = at::cuda::jit::typeName<at::opmath_type<f_inputs_type>>();
      std::string result_type_str = at::cuda::jit::typeName<result_type>();
      auto code = at::cuda::jit::generate_code(nTensors, f, string_name,
                                               f_inputs_type_str, compute_type_str, result_type_str,
                                               /*contiguous=*/true, /*dynamic_casting=*/false,
                                               scalar_pos,
                                               vectorized, vec_size);
      std::string kernel_name = vectorized ? string_name + "_vectorized" + std::to_string(vec_size) : string_name;
      *fn_ptr = at::cuda::jit::jit_pwise_function(code, kernel_name);
    }
  }

  if (vectorized) {
    std::array<void*, 7> args = {
      (void*)&N,
      (void*)&data,
      (void*)&scalar_val,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    };

    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto ic = TrivialOffsetCalculator<arity>();
    auto oc = TrivialOffsetCalculator<1>();
    auto l = memory::LoadWithoutCast();
    auto s = memory::StoreWithoutCast();

    std::array<void*, 7> args = {
      (void*)&N,
      (void*)&data,
      (void*)&ic,
      (void*)&oc,
      (void*)&l,
      (void*)&s,
      (void*)&scalar_val
    };

    at::cuda::jit::launch_jitted_pwise_function(*fn_ptr, args, grid, num_threads());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

}
```

Once we enter the function, we check our cache `fns4/fns2/fns1` to get the pointer to kernel function if it exists. If it does not we use the utilities we looked at above. `generate_code` is called with our computation string to generate the actual kernel code.
`jit_pwise_function` is called to compile and get the function pointer to this kernel. Notice that we use the vectorized kernel if possible in `launch_jitted_vectorized_kernel`. We update our cache pointer to point to this compiled kernel and use `launch_jitted_pwsie_function` from JIT utility to launch with the relevant kernel args and input-output data. Once the kernel is launched, the computation occurs on GPU and user is happy.

On the next run of this operator, the pointer from the cache will be valid and we will directly use it instead of compiling the kernel again.

### Limitations of JITerator

There is a [tracking issue](https://github.com/pytorch/pytorch/issues/69463) to track the limitations and improvements.

### Final Words

JITerator is an interesting technology which solves the problem of increasing CUDA context size and compilation time in PyTorch.
This [post](https://dev-discuss.pytorch.org/t/keeping-pytorchs-ops-maintainable-the-jiterator/468) by Natalia also talks about JITerator and scope of the future work.
