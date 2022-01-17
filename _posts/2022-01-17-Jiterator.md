---
layout: post
title:  "JITerator"
date:   2022-01-17 14:14:49 +0530
categories: pytorch, cuda
---

PyTorch is one of the leading Deep Learning framework. It supports a lot of operations to operate on Tensor.
This rich set of operators allow researchers to quickly proto-type new solutions to their problems. As these operators
are the basic building block of any PyTorch script, it is imperative that they are as performant as possible. Writing performant
operators with parallelization while keeping the cache hot is not easy. Writing these operator considering that the `Tensor`s are not always laid out contiguously in memory, is certainly tricky. Handling these myraid of cases while maintaining performance does not sounds like a non-trivial development task.

### Enter the TensorIterator!
PyTorch internals has a nice and friendly infrastructure to do the same. It helps take care of writing performant code while handling
all these edge cases. So all that the developer needs to add is the actual computation of the operator. When using TensorIterator,
the developer only has to specify the `inputs` and `outputs` for the operator and how to transform those inputs to the output value.
TensorIterator also has extra checks and features to make sure `inputs` and `outputs` don't overlap, they are on same the `device`, they have same the `dtype`. It also takes care of type promotion if the correct flags are set.

Example code using `TensorIterator` (from first blog reference)
```C++
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

For more details on `TensorIterator` one can refer the following blogs which get into the detail of how it works and have various examples
about it's usage.

Blogs:
  * https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update/index.html
  * https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals/

### TensorIterator and CUDA

The existing machinery for using TensorIterator to generate CUDA Kernels of the operation is really easy to use but there is one major caveat in the process. The helper functions like `cpu_kernel` and `gpu_kernel` which generate the kernel code for the operator are heavily templated. These templates are instantiated for all the dtypes that are supported by the operator and if a binary operator supports scalar argument, then two more kernels are instantiated per dtype for `Tensor-Scalar` and `Scalar-Tensor` case. This leads to a lot of increased build time especially for `nvcc` (CUDA AOT compiler). Also these kernels are compiled using CUDA Runtime API, which means that these kernels are loaded when PyTorch binary is loaded which increases the CUDA context size (which ends up consuming actual GPU VRAM). These issues became very apparent while adding new operators for `torch.special` module. A general user who didn't care about these operators had to actually pay the cost (in terms of GPU memory) for them when we loaded PyTorch with CUDA support.

### The JITerator

The solution to these problems that the PyTorch maintainers Mike Rubbery and Natalia Gimelshein came up with was to use NVRTC (Runtime Compilation) library shipped with CUDA to delay the compilation and loading of these kernels. Another way to put it is,
we will not compile the kernel and load the corresponing kernel binary till the operator is called for the first. This way there is not a heavy upfront compilation cost and the operator kernel code is loaded in the GPU memory only if the operator is actually used.

The way this conceptually works is when PyTorch is built it keeps the string representation of the kernel code which needs to be compiled when an operator is called. So when a user actually calls the operator, we check in cache if there is already an kernel available, if not NVRTC is utilized to generate kernel and load it for use. The address of this generated kernel is cached so that next time when is operator is called, we can use this compiled kernel. We will expand this idea in the following section and dive deeper into JITerator.

* How to use JITerator with TensorIterator
    * Computation String (`jiterator_stringify`)
    * Using `jitted_gpu_kernel`
* Diving Deeper
    * `jitted_gpu_kernel`
    * `jitted_gpu_kernel_impl`
    * NVRTC JIT utility helpers
    * `launch_jitted_unrolled_kernel` and `launch_jitted_vectorized_kernel`

#### Computation String
Let's take a look at how the said string looks in code. We will look at the implementation of the binary operator `gcd`. Below is the code for computing `gcd` of two numbers `a_in` and `b_in`.
Do note that we code written doesn't look like string. This is because of the macro `jiterator_stringify` which converts the code to string using preprocessor capabilities. Thanks to this we don't loose out on syntax highlighting and the code still feels like code even if we are actually getting the string of the code in the variable `gcd_string`.
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

#### Using `jitted_gpu_kernel`
Now that we have string representation of our computation, we need to setup the TensorIterator and pass it to the JITerator machinery which will hold onto to this computation string and compile it once the relevant operator is called.

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

`jitted_gpu_kernel` is the entry point for the JITerator. It takes the name of the kernel, return dtype, computation dtype and the number of input arguments as template parameter. It also takes the TensorIterator `iter` and the string for the computation `gcd_string` as run-time argument. At this point we are done with actually implementing the operator. When the operator is called `jitted_gpu_kernel` will take care of compiling the kernel and loading it for use.

### [`jitted_gpu_kernel`](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/Loops.cuh#L111-L113)

As mentioned above, this is the entry point for JITerator. The relevant code is present in `aten/src/ATen/native/cuda/Loops.cuh`.

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

This function has some checks on `input`, `output` and figures out if we need the kernel requires dynamic casting. If the input size larger than one that can be handled by 32-bit indexing, we divide the input and output into 32-bit indexable blocks and call this function recursively. Post this the function calls `jitted_gpu_kernel_impl`.

### [`jitted_gpu_kernel_impl`](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L279-L281)
This function is located in 

Below is the actual C++ code

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

The above function all it does if figure out the memory layout of the input and output tensor and adds extra machinery if they are non-contiguous and if dynamic casting is required (computed in `jitted_gpu_kernel`). Post that it passes the computed data to either `launch_jitted_unrolled_kernel` and `launch_jitted_vectorized_kernel`

### NVRTC JIT utility helpers

Before we look at `launch_jitted_vectorized_kernel` or `launch_jitted_unrolled_kernel`, let's take a quick understanding at the JIT utility that they call.

These utility functions are declared in `aten/src/ATen/native/cuda/jit_utils.h` [file](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/jit_utils.h) and defined in `aten/src/ATen/native/cuda/jit_utils.cu` [file](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/jit_utils.cu)

Important utility helpers to know about
* `generate_code` : This function takes the computation string and wraps it in a `string` with the required machinery of reading from input, passing that data to kernel and writing to the output. It also returns a `string`.

* `jit_vectorized_code_template` and `jit_code_template` : These strings are used to wrap the actual computation string to create the actual kerel. These take care of efficiently loading and storing the input and output tensor and calling our computation code on them.

* `jit_pwise_function`: This function takes in the string generated from `generate_code` and kernel name. It uses `NVRTC` functions to compile the code and load that compiled code into the GPU VRAM and return pointer to the kernel function pointer for calling.

* `launch_jitted_pwise_function` : This function takes the kernel pointer we got from `jit_pwise_function` with the data to be passed to the kernel and launch the kernel using the function provided by cuda-toolkit to launch this runtime compiled kernel.

### [`launch_jitted_unrolled_kernel`](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L126-L135) and [`launch_jitted_vectorized_kernel`](https://github.com/pytorch/pytorch/blob/a383d01774beb112e519ae6a5c560eb402c96a31/aten/src/ATen/native/cuda/CUDALoops.cuh#L179-L186)

Besides the using the machinery of consuming vectorized code these functions do identical task. So we will just look at one of them and interested people can look at the other one.

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

Once we enter the function we check our cache `fns4/fns2/fns1` to get the pointer to kernel function if it exists. If it doesn't we use the utilities we looked at above. `generate_code` is called with our computation string to generate the actual kernel code.
`jit_pwise_function` is called to actually compile and get the function pointer to this kernel. Notice that we use the vectorized kernel if possible in `launch_jitted_vectorized_kernel`. We update our cache pointer with to point to this kernel and use `launch_jitted_pwsie_function` from JIT utility to launch this kernel with the input and output. Once the kernel is launched, the computation happens on GPU and we get the output and user is happy.

On next run of the operator, the pointer from the cache will be valid and we will directly use it instead of compiling the code again.

### Limitations of JITerator

There is a [tracking issue](https://github.com/pytorch/pytorch/issues/69463) to track the limitations and improve them.

We will talk about a few here.

* No math ops on complex dtypes : This means that the operator which support complex data can only be partially JITerated.

* Compile Operator for every individual PyTorch run : Curious and observing reader must have noticed that the cache is a static variable and we don't write the compiled kernel anywhere on the file-system. So every time you load PyTorch and use one of the JITerated operator, you end up compiling them on the first run.

* Can't capture runtime state (supported by `gpu_kernel`) : `gpu_kernel` allows us to capture some runtime state, eg. `int n` passed to `torch.polygamma`. Right now this is not supported by JITerator.
