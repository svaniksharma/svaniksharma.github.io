---
title: 'What does `ggml_cont` actually do?'
slug: ggml_cont
date: 2025-05-30T07:07:07+01:00
math: 'mathjax'
---

The following [gist](https://gist.github.com/svaniksharma/15fa1b0dbd0aca853f6e6bef0011bdb0)  will be helpful for the latter part of this article:

## Inspecting `ggml_cont`

Recently, I've been playing around with GGML. While doing so, I was looking through the examples, and I saw [this](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/examples/mnist/mnist-common.cpp#L362-L364)  in `mnist_common.cpp`:

```cpp
dense_in = ggml_reshape_2d(model.ctx_compute,
            ggml_cont(model.ctx_compute, ggml_permute(model.ctx_compute, dense_in, 1, 2, 0, 3)),
            (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), model.nbatch_physical);
```

This was on line 362. It preceded a dense matrix multiplication and addition for a fully-connected layer. It's pretty clear what `ggml_reshape_2d` does. `ggml_permute` was a little confusing at first, but I found [this](https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array#32034565) article that discusses an analogous operation in NumPy that explains what the permutation does. However, `ggml_cont` was a little bit confusing. In `ggml.h`, all [it says is](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/include/ggml.h#L1237-L1240):
```cpp
// make contiguous
GGML_API struct ggml_tensor * ggml_cont(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);
```
Ok, that's a little vague, but it basically tells us that `ggml_cont` makes the supplied tensor contiguous in memory. Let's dig into the code. Looking at `ggml.c`, `ggml_cont` just calls `ggml_cont_impl` which does the [following](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/src/ggml.c#L3012-L3022): 
```cpp
static struct ggml_tensor * ggml_cont_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);


    result->op     = GGML_OP_CONT;
    result->src[0] = a;


    return result;
}
```
Well, this doesn't explain much. But, it shows us that it duplicates the tensor argument `a` and then marks this operation as `GGML_OP_CONT`. Recall that in `mnist_common.cpp`, this is called before `ggml_reshape_2d`. Let's [look](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/src/ggml.c#L3109-L3125)  at that function:
```cpp
struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);


    const int64_t ne[2] = { ne0, ne1 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);


    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;


    return result;
}
```

This function has a precondition that `a` must be contiguous, which it checks with `ggml_is_contiguous`. Looking through a sequence of nested calls, we see that `ggml_is_contiguous_n` is the function that's called: Now, let's look at [that](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/src/ggml.c#L1303-L1323):
```cpp
static bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}
```
Reading through this, we can see finally what is happening. GGML checks if every dimension of order `n` or above is contiguous (here, `tensor->nb` is the "stride" and `tensor->ne` holds the shape of `tensor`). The function `ggml_is_contiguous` effectively calls `ggml_is_contiguous(tensor, 0)`, basically checking that adjacent elements are, in fact, contiguous in memory. 

## Using `ggml_cont` in a toy program

You might think this seems a little arduous. From the first comment above `ggml_cont`'s declaration that said "make contiguous", you could surmise that, indeed, `ggml_cont` made all the adjacent elements contiguous. But, now we examine `ggml_cont`'s behavior (the way I did it the first time before I looked at the source code). I wrote a little program, and inside of `main`, I had this:
```cpp
struct ggml_init_params params = {
    1024 * ggml_tensor_overhead(),
    nullptr,
    false
  };
  struct ggml_context *ctx = ggml_init(params);
  // Creates an array with values 0, 1, 2, ..., 15
  const int N = 16;
  int values[N] = { 0 };
  for (int i = 0; i < N; i++)
    values[i] = i;
  struct ggml_tensor *tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
  for (int i = 0; i < N; i++)
    ggml_set_i32_1d(tensor, i, values[i]);
  struct ggml_tensor *t = ggml_reshape_4d(ctx, tensor, 2, 2, 4, 1);
  std::cout << "Original tensor\n--------------\n";
  print_tensor(t);
  // 0 -> 1, 1 -> 2, 2 -> 0, 3 -> 3
  struct ggml_tensor *permuted_t = ggml_permute(ctx, t, 1, 2, 0, 3);
  std::cout << "Permuted tensor\n--------------\n";
  std::cout << "New Shape: " << permuted_t->ne[0] << " x " << permuted_t->ne[1] << " x " << permuted_t->ne[2] << " x " << permuted_t->ne[3] << "\n";
  GGML_ASSERT(permuted_t->ne[0] == 4);
  GGML_ASSERT(permuted_t->ne[1] == 2);
  GGML_ASSERT(permuted_t->ne[2] == 2);
  GGML_ASSERT(permuted_t->ne[3] == 1);
  print_tensor(permuted_t);
```
Note: `print_tensor` is available at the gist at the start of the article. It does exactly what you think it does.
This doesn't do anything with `ggml_cont` just yet. It initializes an array with integers between 0 and 15, inclusive. Then, it stores it in a GGML tensor. I then reshape `tensor` to get a $2 \times 2 \times 4 \times 1$ tensor called `t`. Then, I permute the dimensions of this tensor to get a $4 \times 2 \times 2 \times 1$ tensor called `permuted_t`. This is pretty simple, so let's look at the output:
```shell
Original tensor
--------------
0 1 
2 3 

4 5 
6 7 

8 9 
10 11 

12 13 
14 15 


Permuted tensor
--------------
New Shape: 4 x 2 x 2 x 1
0 4 8 12 
1 5 9 13 

2 6 10 14 
3 7 11 15 
```

## An unexpected turn

So far, so good. Now, I apply `ggml_cont` and then reshape the tensor to be $8 \times 2$:
```shell
struct ggml_tensor *cont_permuted_t = ggml_cont(ctx, permuted_t);
struct ggml_tensor *a = ggml_reshape_2d(ctx, cont_permuted_t, 2, 8);
GGML_ASSERT(a->ne[0] == 2); 
GGML_ASSERT(a->ne[1] == 8);
GGML_ASSERT(a->ne[2] == 1);
GGML_ASSERT(a->ne[3] == 1);
print_tensor(a);
```

Based on what we saw before, this should give me something like:
```shell
0 4 
8 12 
1 5 
9 13 
2 6 
10 14 
3 7 
11 15
```

But, in reality, we get this:
```shell
0 0 
0 0 
0 0 
0 0 
0 0 
0 0 
0 0 
0 0 
```

What's going on? We called `reshape_4d` and it was fine, so clearly the issue is with `ggml_cont`. I was stumped for a bit, so I asked DeepSeek. It gave me some verbose output, most of which was useless. But there was one part of the response that solved half the puzzle:
> Tensor not actually computed yet: GGML uses a graph-based approach where operations are only computed when needed.

Of course! When you create a tensor in GGML (like when you call `ggml_new_tensor_1d`), the tensor does not hold any data. You are building the computational graph, i.e, the sequence of operations that you want GGML to perform. To actually *compute* the answer, you need to allocate the computational graph as follows:
```cpp
struct ggml_cgraph *gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, a);
ggml_graph_compute_with_ctx(ctx, gf, 1);
// Try printing the tensor
std::cout << "Now-contiguous tensor after we perform the computation\n";
print_tensor(a);
```
Then you get the desired output:
```shell
Now-contiguous tensor after we perform the computation
0 4 
8 12 
1 5 
9 13 
2 6 
10 14 
3 7 
11 15 
```

## Glass half empty

But this still leaves one problem. How come `ggml_reshape_4d` and `ggml_permute` worked *before* we allocated the computational graph? This is where our inspection of the source code pays off: if you look at `ggml_reshape_2d` again:
```cpp
struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);


    const int64_t ne[2] = { ne0, ne1 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(result, "%s (reshaped)", a->name);


    result->op     = GGML_OP_RESHAPE;
    result->src[0] = a;


    return result;
}
```

You can see that `ggml_new_tensor_impl` is called, and the tensor `a` is passed to it. Therefore, when reshaping, the data in `a` **is passed** to result. So, when we get the result of `reshape_4d` (or `reshape_2d` or `reshape_3d`), it contains the original data from `a`, with the only difference being the shape of the tensor. This explains why we are able to print the result of `reshape_4d`. A similar situation holds for `ggml_permute`, except it calls `ggml_view_tensor` on `a`.
Now, let's look at `ggml_cont` again:
We see that 
```cpp
static struct ggml_tensor * ggml_cont_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
    ggml_format_name(result, "%s (cont)", a->name);


    result->op     = GGML_OP_CONT;
    result->src[0] = a;


    return result;
}
```
We see that it calls `ggml_dup_tensor`, which is implemented [as follows](https://github.com/ggml-org/ggml/blob/62042b741f0a7ac8c5a33d8d98129a9dcb6bbdd9/src/ggml.c#L1698-L1700):
```cpp
struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}
```

So this calls `ggml_new_tensor`, but this time `src` **is not passed** in. Indeed, `ggml_new_tensor` just creates a new tensor of the same type and same dimensions as `src`, but it does not supply any of the data contained in `src` itself. The result is that we get an empty tensor with the same dimensions as `src` (whose adjacent elements *are* contiguous in memory), but it does not contain any data. This solves the second half of the puzzle: we didn't allocate the computational graph **and** we cannot trust -- without inspecting the source code -- that an operation like `ggml_cont`, `ggml_reshape_2d`, or `ggml_permute` will execute the desired transformation **before** the computational graph is allocated and computed. 

## Conclusion

**TLDR**: Allocate the computational graph **before** inspecting the result of any tensor operations, whether that be `ggml_cont`, `ggml_conv_2d`, `ggml_pool_2d`, etc. Though some of these operations (like reshaping) will preserve the original data, some of them do not execute until the computational graph is built and allocated.
