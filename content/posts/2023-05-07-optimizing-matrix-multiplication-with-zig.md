+++
title = "Optimizing Matrix Multiplication with Zig"
date = 2023-05-07T07:07:07+01:00
draft = false
+++

I recently started playing with the [Zig](https://ziglang.org/) programming language and wanted to try it out for its speed.
And what better way to do that than to try optimizing matrix multiplication? Since there are a plethora of resources to 
understand how to multiply matrices efficiently (see the Resources section below), I won't be doing anything intense in this article
(though maybe in the future I will).

The naive matrix multiplication algorithm is given below in Zig:

```zig
fn naiveMatrixMultiply(C: anytype, A: anytype, B: anytype) void {
    const N = A.len;
    for (0..N) |i| {
        for (0..N) |j| {
            for (0..N) |k| {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
```

We'll iteratively optimize this, using `perf` to benchmark our findings. At the bottom of this post is the boiler plate I used to test the code's
correctness, how I generated the matrices, etc. The optimizations done here are applied to square matrices and all `perf` output shows the time
taken to multiply two randomly-generated 1000 x 1000 matrices. The following is the output of `perf stat -e cache-misses,cache-references,instructions,cycles ./matrix` after compiling `zig build-exe matrix.zig`
and running `naiveMatrixMultiply`:

```shell
1.5698057476e+04 ms

 Performance counter stats for './matrix':

       127,513,715      cache-misses              #   96.317 % of all cache refs
       132,389,208      cache-references
    87,573,830,478      instructions              #    1.91  insn per cycle
    45,758,041,155      cycles

      15.928915857 seconds time elapsed

      15.899428000 seconds user
       0.023993000 seconds sys
```

In the original code, I measure the time taken to execute the function and then print it out, which is why there is a `1.56e+04 ms` at the top of the output. We won't be using this for benchmarking, however.
Unless otherwise stated, all `perf` output is the result of running `zig build-exe matrix.zig` and then running `perf stat -e cache-misses,cache-references,instructions,cycles ./matrix`.

## Optimization #1: Transpose the matrix

Notice that matrix `B` is iterated over in column-major order. That is, we iterate over the elements of `B` like so: (0, 0), (1, 0), (2, 0), ..., (N-1, 0), (0, 1), etc.
Notice that (0, 0) and (0, 1) are in the same cache line. Therefore, by transposing `B`, we can ensure that we are traversing both matrices in row-major order, which allows us to hit the cache more often.
```zig
fn transposeMatrixMultiply(C: anytype, A: anytype, B: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var tmp: [][]f64 = try allocator.alloc([]f64, B.len);
    for (tmp) |*row| {
        row.* = try allocator.alloc(f64, B.len);
    }
    for (0..B.len) |i| {
        for (0..B.len) |j| {
            tmp[i][j] = B[j][i];
        }
    }
    for (0..B.len) |i| {
        for (0..B.len) |j| {
            for (0..B.len) |k| {
                C[i][j] += A[i][k] * tmp[j][k];
            }
        }
    }
}
```

Our `perf stat` output for this new function is:
```shell
1.0450551452e+04 ms

 Performance counter stats for './matrix':

         4,606,282      cache-misses              #   65.597 % of all cache refs
         7,022,066      cache-references
    91,634,171,605      instructions              #    2.99  insn per cycle
    30,664,602,948      cycles

      10.680824428 seconds time elapsed

      10.665631000 seconds user
       0.011997000 seconds sys
```

Compared to the naive output, this is pretty good. We are definitely hitting the cache more often and, in terms, of cycles,
we see a 33% speedup! However, we can do even better.

## Optimization 2: SIMD + Transpose

In addition to the transpose, we can try to use SIMD (Single-Instruction, Multiple-Data) instructions. If we were programming this in C,
we would have to use SIMD intrinsics, which are not only sometimes difficult to use but not very portable. However, Zig offers
the `Vector` datatype, which allows one to operate on multiple elements at the same time. The code now looks like this:
```shell
fn transposeSimdMatrixMultiply(C: anytype, A: anytype, B: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var tmp: [][]f64 = try allocator.alloc([]f64, B.len);
    for (tmp) |*row| {
        row.* = try allocator.alloc(f64, B.len);
    }
    for (0..B.len) |i| {
        for (0..B.len) |j| {
            tmp[i][j] = B[j][i];
        }
    }
    const vec_len = 32;
    for (0..B.len) |i| {
        for (0..B.len) |j| {
            var k: usize = 0;
            while (k <= B.len - vec_len) : (k += vec_len) {
                const u: @Vector(vec_len, f64) = A[i][k..][0..vec_len].*;
                const v: @Vector(vec_len, f64) = tmp[j][k..][0..vec_len].*;
                C[i][j] += @reduce(.Add, u * v);
            }
            while (k < B.len) : (k += 1) {
                C[i][j] += A[i][k] * tmp[j][k];
            }
        }
    }
}
```

It's a bit longer, but there's nothing too bad here. The innermost loop now operates on `vec_len = 32` elements at a time,
multiplying sets of 32 elements in each row of `A` and `tmp` and then summing the elementwise products together. If the number
of elements left at the end of the loop isn't a multiple of 32, then we revert back to the same algorithm as the `transposeMatrixMultiply` function. 
Here's the `perf` output:
```shell
2.319826094e+03 ms

 Performance counter stats for './matrix':

         5,522,065      cache-misses              #   21.862 % of all cache refs
        25,258,801      cache-references
    12,249,595,729      instructions              #    1.84  insn per cycle
     6,672,691,818      cycles

       2.546859528 seconds time elapsed

       2.538763000 seconds user
       0.008008000 seconds sys
```
Again, a substantial decrease. We're now operating at 6 billion cycles, or 14% of the number of cycles taken by the naive function.
Also, a much smaller proportion of our cache references are cache misses. Compared to working with SIMD intrinsics by hand, this
definitely is a lot of power at your fingertips. In the [AVX2 instruction set](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) (which is 
the instruction set my machine uses), each vector register is 256 bits and there are 8 registers, but if you are using intrinsics, you
would have to manage these 8 registers separately. However, the generic interface provided by `Vector` means that we can treat these 8
registers as one big register, each containing 4 64-bit floating point numbers, allowing us to operate on 32 elements at a time! 

## Optimization 3: SIMD with unrolled loop

Practically, we cannot always afford to transpose a matrix. Besides the `O(N^2)` runtime to actually transpose it, we also take up extra memory.
For the next optimization, we will use SIMD, but we won't transpose the `B` matrix. Going back to what we observed before, the access pattern
for `B` is suboptimal: when we access `B[k][j]`, we access `B[k][j+1]` in the inner loop only after *another* round of the inner loop. However,
that if we did use it while we were in the inner loop. That is, while we computed `C[i][j] += A[i][k] * B[k][j]`, we also computed `C[i][j+1] += 
A[i][k] * B[k][j+1]`? Since `i` doesn't change until the `j`-loop is over and `k` only changes after we are done with `C[i][j] += A[i][k] * B[k][j]`,
we can take advantage of the moment we have access to the elements in the slice `B[k][j..]` and use it to compute the elements that will belong in 
`C[i][j..]`. The following code puts this thought into action:
```zig
fn unrollSimdMatrixMultiply(C: anytype, A: anytype, B: anytype) void {
    const N = B.len;
    const vec_len = 32;
    for (C, A) |*C_row, *A_row| {
        var j: u32 = 0;
        while (j <= N - vec_len) : (j += vec_len) {
            for (0..N) |k| {
                const u: @Vector(vec_len, f64) = B[k][j..][0..vec_len].*;
                const y: @Vector(vec_len, f64) = C_row.*[j..][0..vec_len].*;
                const w: @Vector(vec_len, f64) = @splat(vec_len, A_row.*[k]);
                const slice: [vec_len]f64 = (u * w) + y;
                @memcpy(C_row.*[j .. j + vec_len], &slice);
            }
        }
        while (j < N) : (j += 1) {
            for (0..N) |k| {
                C_row.*[j] += A_row.*[k] * B[k][j];
            }
        }
    }
}
```
Note that I replaced the `i` loop and decided to loop over the rows of `C` and `A` directly. What we are doing is again straightforward. 
We just take `vec_len = 32` elements `B[k][j], B[k][j+1], ..., B[k][j + 31]`, multiply them by `A[i][k]` (which is now `A_row.*[k]`), and then 
store it in `C[i][j], C[i][j+1], ..., C[i][j + 31]` (which is now `C_row.*[j], C_row.*[j+1], ..., C_row.*[j+31]`). Again, if we have less than 32
elements remaining, we revert back to the standard multiplication algorithm. As always, the `perf` output is below:
```shell
5.233718283e+03 ms

 Performance counter stats for './matrix':

       101,785,707      cache-misses              #   63.052 % of all cache refs
       161,432,535      cache-references
    16,377,067,907      instructions              #    1.15  insn per cycle
    14,227,983,666      cycles

       5.462324961 seconds time elapsed

       5.457798000 seconds user
       0.004001000 seconds sys
```

Compared to our previous output, this isn't great. However, still a significant improvement from our naive and only-transpose matrix
multiplication functions. The main issue here is the `k`-loop: though we are leveraging `Vector` to use nearby data, we are still missing
the cache (and in fact our proportion of cache-misses in relation to cache-references is similar to the only-transpose matrix function). 
Still we are not using up extra memory, which is a good bonus. However, there is one more optimization we can do.

## Optimization 4: Compilation Arguments

By default, Zig is in the Debug build mode, which means that it enables all runtime safety checks with no optimizations. However, we can
change this build mode by running `zig build-exe -O ReleaseFast ./matrix` (which builds without runtime-safety checks and optimizes for speed).
Now running `perf` on the unrolled SIMD loop, we get the following:
```shell
1.636596222e+03 ms

 Performance counter stats for './matrix':

       116,986,587      cache-misses              #   70.848 % of all cache refs
       165,123,281      cache-references
     1,133,618,990      instructions              #    0.29  insn per cycle
     3,970,347,473      cycles

       1.652725296 seconds time elapsed

       1.640017000 seconds user
       0.012000000 seconds sys
```

With quite literally zero coding effort or thinking at all, we have beaten our previous record! To be fair, if you run `zig build-exe -O ReleaseFast ./matrix`
using the `transposeSimdMatrixMultiply` function, you will find that it is still faster with optimizations as well. However, considering we were trying to avoid
transposes and put in minimal effort, I would say this level of optimization is pretty good. Another thing I should note is that a significant proportion of the
time is taken with the builtin `@memcpy` function. Running `perf record -a ./matrix` in Debug mode and then looking at `perf report` gives me this output:
```
 81.08%  matrix           matrix                               [.] matrix.unrollSimdMatrixMultiply__anon_3611                            
   9.86%  matrix           matrix                               [.] memcpy                                                                
   1.46%  swapper          [unknown]                            [k] 0xffffffffb372cdaf                                                    
   1.42%  matrix           matrix                               [.] rand.Xoshiro256.fill
... rest of output omitted
```
I also tried using `std.mem.copy`, but it was actually worse than the builtin function. However, the reason I used `@memcpy` was because there doesn't seem to be another choice.
There was no "store" function I could use. If we were in C, we could have just used something like `_mm256_store_pd` to store the data into `C`, but the `Vector` datatype
does not seem to have anything like that. However, I think the `Vector` interface is still being worked on, so it's possible this will be ironed out in later versions.

## Conclusion

In the resources section below, I provided some links to some good material I found on matrix multiplication and how to optimize it. If you look at them, you'll find
that it can get quite involved really quickly. However, most of the optimizations are related to improving cache hits. Furthermore, what I did in this article is by no means
the limit, and especially not with Zig. I'm sure if you dug into the `@memcpy` function, figured out how to use SIMD intrinsics within Zig, or used some of the other builtin
functions (like `@preFetch`, which sounds quite useful in this case), you can further optimize what I wrote. Plus, I'm a complete beginner to Zig, so I'm pretty sure a lot of
what I wrote was suboptimal to some degree. Nevertheless, I'm quite optimistic to see a performance-oriented language like this, and being able to optimize a complex problem
like this very quickly is extremely promising. The full code is available [here](https://gist.github.com/svaniksharma/9ad2fa148254ac74b02940326090b18d).

## Resources

* [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) 
* [Matrix Multiplication](https://en.algorithmica.org/hpc/algorithms/matmul/)
* [Matrix multiplication algorithm](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
