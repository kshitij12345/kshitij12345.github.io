---
layout: post
title:  "GPU Parallel Programming: Chapter 1"
date:   2021-08-07 14:14:49 +0530
categories: c++, cuda
---

### Chapter 1: Introduction to CPU Parallel Programming

The promise of Moore's Law is dead (though not the Moore's Law itself). The number of transistor's on a chip are
increasing every 2 years, but the frequency increase has hit wall due to power consumption and heating issues.
So these new chips have more cores and for now it's upto the programmers to get the best out of these multi-core systems.

#### More core doesn't directly mean more performance

It is imperative to understand the fact that just because you make your program parallel doesn't mean, it will be faster.
It is important to correctly orchestrate the work threads are doing. Also, with the fact the memory access being slower,
sometimes your program can get memory bound i.e. your cores are being underutilized and waiting for the data to be ready.
So parallel programming is mmore than just throwing more threads at your problem (unless it's embarassingly parallel).

#### Data-bandwith for different devies

* Network Interface Card : 1Gbps (Gigabits per second)
* HDD connected over PCI3 bus : 1-2 Gbps (6Gbps max possible)
* USB 3 : Max 10Gbps
* SSD over PCI3 bus : 4-5 Gbps (6Gbps max possible)
* RAM : 20-60 GBps (Gigabytes per second) / 160-480 Gbps
* GPU Internal Memory : 10-1000 GBps

#### Interesting Note from the book

We can get less noisy performance/benchmark data if our data fits in cache because once data is in cache the execution
is fairly deterministic. However if data spills over the cache, then due to non-deterministic nature of memory access
from RAM (due to other programs running on OS and OS overheads) the benchmark data will be more noisy!

