---
layout: post
title:  "Gentle Introduction to Valgrind!"
date:   2021-04-29 14:14:49 +0530
categories: intro
---

## Valgrind
[Valgrind](https://www.valgrind.org/downloads/) is an useful tool if you are working with C/C++. It is a suite of multipurpose tools with varied functionality. It allows you to detect memory-leak, profile your code and more. One of the thing that it shines at is helping debugging trick memory bug.

### How does it work?
It is worth noting that Valgrind is language agnostic and works with all languages, compiled or interpreted. Since Valgrind consumes a binary, it does not care about which language it came from. Valgrind is `non-intrusive` in terms of adding an instrumentation code. You need not update your codebase for it to work with Valgrind. The way it manages to do this is by providing a virtual core for the binary to run on. This way it can also figure out statistics like number of number native instructions executed.

### Installing Valgrind
Let's start with setuping up a playground environment with Valgrind
```bash
$conda create -n valgrind-test-env
$conda install -c conda-forge compilers
$conda install -c conda-forge valgrind
```

### Example Program
```cpp
int main()
{
    auto x = new char[10];
    delete x;
    return 0;
}
```

This innocuous looking program is actually ill-formed. The reason being, with `new[]` allocation the allocated memory should be freed with `delete[]`, with `delete` the behavior is undefined.

### Building the program
Let's build with most of the compiler warnings enabled to see if our compiler can foresee this issue and warn us.

```bash
g++ example.cc -Wall -Wpedantic -Werror -Wextra
```

On running the following, we get a shiny new executable without any warnings.

### Running under Valgrind
```
$valgrind --tool=memcheck ./a.out
```

Here we are asking Valgrind to run `memcheck` on our binary. There are more tools and options which can be specified.

On running the above command, we get the following output.
```
==19750== Memcheck, a memory error detector
==19750== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==19750== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==19750== Command: ./a.out
==19750== 
==19750== Mismatched free() / delete / delete []
==19750==    at 0x403713B: operator delete(void*, unsigned long) (vg_replace_malloc.c:595)
==19750==    by 0x1091AD: main (in /home/user/Desktop/Repositories/valgrind-test/a.out)
==19750==  Address 0x51ccc80 is 0 bytes inside a block of size 10 alloc'd
==19750==    at 0x40365AF: operator new[](unsigned long) (vg_replace_malloc.c:433)
==19750==    by 0x109193: main (in /home/user/Desktop/Repositories/valgrind-test/a.out)
==19750== 
==19750== 
==19750== HEAP SUMMARY:
==19750==     in use at exit: 0 bytes in 0 blocks
==19750==   total heap usage: 2 allocs, 2 frees, 72,714 bytes allocated
==19750== 
==19750== All heap blocks were freed -- no leaks are possible
==19750== 
==19750== For lists of detected and suppressed errors, rerun with: -s
==19750== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```

Note the `Mismatched free() / delete / delete []`. The message tells us that we have a mismatched `free() / delete / delete []` for the `new[]`. Valgrind is able to detect this as it does the booking keeping of the instruction it executed, instructions which requested memory and instructions which freed it.

## Caveats
Since Valgrind emulates the hardware and does more tracking and bookkeeping, running a program under Valgrind is much slower.

Valgrind's site states
```
So what's the catch? The main one is that programs run significantly more slowly under Valgrind. Depending on which tool you use, the slowdown factor can range from 5--100. This slowdown is similar to that of similar debugging and profiling tools. But since you don't have to use Valgrind all the time, this usually isn't too much of a problem. The hours you'll save debugging will more than make up for it.
```

## Epilogue

All in all, Valgrind is worthwhile tool to catch bugs. Even though it is slow, it will cut down on the debugging time by a large margin.
