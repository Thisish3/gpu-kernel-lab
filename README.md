# GPU Kernel Lab

A hands-on study repository for understanding and implementing high-performance GPU kernels used in modern AI inference systems.

## Objectives

* Learn Triton kernel programming from fundamentals
* Understand GPU memory hierarchy and pointer arithmetic
* Implement optimized kernels such as Vector Add, Softmax, LayerNorm, and GEMM
* Analyze production-grade kernels from Triton, FlashAttention, and vLLM
* Build a portfolio demonstrating GPU kernel optimization skills

## Learning Roadmap

* GPU Architecture
* Memory Hierarchy
* Pointer Arithmetic
* Tiling
* GEMM
* Softmax
* LayerNorm
* FlashAttention
* vLLM Kernel Analysis
* PyTorch Custom Extensions

## Repository Structure

```text
docs/        # Learning notes and architecture explanations
triton/      # Triton kernel implementations
pytorch/     # PyTorch extensions and experiments
analysis/    # Production kernel analysis
benchmark/   # Performance benchmarking
images/      # Figures and diagrams
notebooks/   # Experimental notebooks
```

## Progress

* [x] Environment setup
* [x] Triton Vector Add
* [x] Pointer arithmetic
* [x] Tile indexing
* [ ] Softmax kernel
* [ ] GEMM optimization
* [ ] FlashAttention analysis
* [ ] vLLM kernel analysis
* [ ] PyTorch custom extension

## References

* Triton Language Documentation
* PyTorch Documentation
* FlashAttention
* vLLM
