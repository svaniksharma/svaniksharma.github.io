+++
title = 'A Short GGML Tutorial'
date = 2025-05-28T07:07:07+01:00
draft = false
math = 'mathjax'
+++

I recently wanted to learn about [GGML](https://github.com/ggml-org/ggml), a tensor and machine learning library behind popular open source versions of [LLaMA](https://github.com/ggml-org/llama.cpp) and [Whisper](https://github.com/ggml-org/whisper.cpp). Trying to find good resources on GGML is hard, so I thought I'd write up some preliminary notes for anyone looking to get started. 

Before reading the rest of this, please consider the following resources, especially the first link -- the examples in the GGML repository are a great starting point:
* [GGML Examples](https://github.com/ggml-org/ggml/tree/master/examples), especially:
    - [Simple Example w/ Static Context](https://github.com/ggml-org/ggml/blob/master/examples/simple/simple-ctx.cpp) 
    - [Simple Example w/ Backend](https://github.com/ggml-org/ggml/blob/master/examples/simple/simple-backend.cpp)
    - [MNIST Implementation](https://github.com/ggml-org/ggml/tree/master/examples/mnist)
* [Huggingface Article](https://huggingface.co/blog/introduction-to-ggml)
* [Deep Dive into GGML](https://xsxszab.github.io/posts/ggml-deep-dive-ii/) (this is pretty advanced since it discusses GGML internals, too)

## Tutorial Regression

In order to get started with GGML, our goal will be to implement a simple program that can perform a linear regression. In particular, we will fit the function:

$$ y = ax + b $$

given data points $(x_i, y_i)$, $i \in {1, ..., n}$. All the code presented henceforth is available [here](https://github.com/svaniksharma/ggml-tutorial). We start off in `main` doing the [following](https://github.com/svaniksharma/ggml-tutorial/blob/74bdf6a8d231bf21efb73b21b592515bd22cda0b/src/tutorial.cpp#L83C3-L87C51):
```cpp
  /* TutorialRegression: Compute a * x + b where a = 3, b = 4, x = 5, (end result should be 19) */
  TutorialRegression regressor;
  regressor.set_params(3.0f, 4.0f);
  float result = regressor.forward(5.0f);
  std::cout << "Tutorial Result: " << result << "\n";
```
The `TutorialRegression` object is used to compute $ax + b$ where $a = 3, b = 4, \text{ and } x = 5$. This object is something we create to encapsulate the regression process and, in particular, we are only going to use it to do inference (so no training). The internals are written in GGML, which we'll dive into right now, starting with the [constructor](https://github.com/svaniksharma/ggml-tutorial/blob/74bdf6a8d231bf21efb73b21b592515bd22cda0b/include/tutorial.h#L15C3-L34C4) in `tutorial.h`:
```cpp
  TutorialRegression() {
    struct ggml_init_params params {
      /* .mem_size = */ 1024 * ggml_tensor_overhead(), 
      /* .mem_buffer = */ nullptr,
      /* .no_alloc = */ false
    };
    /* We initialize a context. 
     * The context keeps track of the tensors and operations between them. 
     * In this case, we are also using it to perform the computation. */
    _ctx = ggml_init(params); 
    // Using the context, we construct the computational graph for y = a * x + b.
    // Note that no computation is actually being performed.
    // This is only done to build the graph. 
    _a = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _b = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    _x = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *ax = ggml_mul(_ctx, _a, _x);
    _result = ggml_add(_ctx, ax, _b);
  }
```
This class has some private variables, which you can identify by the underscore prefix. The first line initializes `params`, which contains three fields of importance to us: `mem_size`, `mem_buffer`, and `no_alloc`. `mem_size` specifies the amount of memory to allocate for the following operations. Though we could try to calculate it, we will just allocate `1024 * ggml_tensor_overhead()`, i.e, the amount of memory needed to allocate 1024 tensors. This is almost certainly too much, but it suffices for this example. `mem_buffer` allows us to specify a memory buffer for GGML to use, but we can set it to null and let GGML take care of finding the memory for us. Finally, `no_alloc` is used to tell GGML whether allocate memory for the computational graph or not. In this case, we have set it to `false`, meaning we want GGML to allocate the memory for the computational graph for us. This will be important later.

After initializing `params`, we pass it to `_ctx`. This creates a `ggml_context`, which is used to track tensor metadata and will help us build the computational graph. Using `_ctx`, we initialize `_a`, `_b`, and `_x` as new tensors (in this case, the "tensors" are just 32-bit floats). Note that we have not actually allocated any memory nor have we initialized the tensors with any particular value. At this point, the tensors are just objects in a computational graph. The next few lines make this clear. We "multiply" `_a` by `_x` and then we "add" `_b` to get `_result`. Using the `_ctx`, this creates a computational graph that tells GGML how to compute $y = ax + b$ (in this case, $y$ is `_result`).

Now, we can move into `tutorial.cpp`. The next line after creating the `TutorialRegression` object in `main` is `regressor.set_params(3.0f, 4.0f);`. We can now examine the `set_params` [method](https://github.com/svaniksharma/ggml-tutorial/blob/e82d5ff8560c1d65fd3813bfe5544b215025f53c/src/tutorial.cpp#L10-L15):
```cpp
void TutorialRegression::set_params(const float a, const float b) {
  // This just sets the values of `_a` and `_b`. This is used once we are 
  // ready to perform the computation (see `forward`).
  ggml_set_f32(_a, a);
  ggml_set_f32(_b, b);
}
```
This is where we actually fill the tensors with some data. Since we are just using the CPU, we can call `ggml_set_f32` to set the data (this is not the same if we are using a GPU or some other "backend"). 
Now, for the interesting part. This is the line `float result = regressor.forward(5.0f);` in `main`. This is what the `forward` [method](https://github.com/svaniksharma/ggml-tutorial/blob/e82d5ff8560c1d65fd3813bfe5544b215025f53c/src/tutorial.cpp#L17C1-L28C2) does:
```cpp
float TutorialRegression::forward(const float x) {
  // Set the input tensor `_x`
  ggml_set_f32(_x, x);
  // Create a new graph using the context
  struct ggml_cgraph *cf = ggml_new_graph(_ctx);
  // Use the new graph and the output tensor to build the graph outwards
  ggml_build_forward_expand(cf, _result);
  // Use the graph and context to compute the result (use 1 thread)
  ggml_graph_compute_with_ctx(_ctx, cf, 1);
  // get the result from the tensor (since this tensor holds 1 value, we pass index 0).
  return ggml_get_f32_1d(_result, 0);
}
```

Like in `set_params`, we fill the input tensor `_x` with some actual data. To actually *do* the computation, we create a computational graph from `_ctx` called `cf`. We then tell GGML to build the graph out and put the result in `_result` by calling `ggml_build_forward_expand`. Finally, we use `ggml_graph_compute_with_ctx` to actually compute the result. Note that `_result` is a tensor, so in order to extract the floating point data, we use `ggml_get_f32_1d`.

If you run this, you should notice the line `Tutorial Result: 19` in the output.

## Backend Regression

Obviously, GGML wouldn't be very useful if we could only compute on the CPU. We are also able to use the GPU and, more generally, different *backends* in order to do inference. Furthermore, you can train models with GGML to. We will do both to complete our regression example. First, we'll look at inference (as we did with `TutorialRegression`) and then we'll look at training.

### Inference

In `main`, we do the [following](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/src/tutorial.cpp#L88C3-L92C53):
```cpp
  /* Do the same thing as TutorialRegression, but with a backend (end result should be 19) */
  BackendRegression<float> backend_regressor;
  backend_regressor.set_params(3.0f, 4.0f);
  result = regressor.forward(5.0f);
  std::cout << "Backend result: " << result << "\n";
```

Similar to `TutorialRegression`, `BackendRegression` encapsulates all the GGML stuff. This is performing the exact same calculation as before, but we will now use the GPU to perform the computation (you will need CUDA if you want to do this). Let's start with the [constructor](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/include/tutorial.h#L89C3-L153C4):
```cpp
  BackendRegression() {
    // This can work for either double or float.
    // Notice that .no_alloc is true here. We want to allocate memory explicitly.
    struct ggml_init_params params = {
      /* .mem_size = */ 1024 * ggml_tensor_overhead(),
      /* .mem_buffer = */ nullptr,
      /* .no_alloc = */ true
    };
    // Like before, we determine whether we are dealing with float or double.
    enum ggml_type tensor_type;
    if (std::is_same<T, float>::value)
      tensor_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      tensor_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    // We now allocate a context, but we will be using a backend for computations. 
    // In this case, the only purpose of the *static* context is to create the tensor metadata.
    _ctx_static = ggml_init(params);
    _a = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _b = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    _x = ggml_new_tensor_1d(_ctx_static, tensor_type, 1);
    // Since we are going to be using this for both training and inference, we need to specify the 
    // model inputs, outputs, and parameters. In y = a * x + b, a and b are parameters, x is an input.
    ggml_set_input(_x);
    ggml_set_param(_a);
    ggml_set_param(_b);
    // Now we initialize the backend. In this case, we use CUDA as the backend.
    // The backend buffer is returned after we allocate the tensors using the static context + backend.
    // We will need to keep the backend buffer to free it later.
    _backend = ggml_backend_cuda_init(0);
    _backend_buffer = ggml_backend_alloc_ctx_tensors(_ctx_static, _backend);
    // Now, we create the *compute* context. This is what does inference and training. 
    // Again, .no_alloc = true because we will explicitly allocate the graph. Calculating the memory needed is easy
    // and we don't have to allocate a sufficiently large amount or check with ggml_mem.
    // By default, GGML allocates 2048 nodes (GGML_DEFAULT_GRAPH_SIZE) when allocating a graph.
    // Each of the 2048 nodes carries overhead since they are essentially tensors. Since we are 
    // doing inference and training, we need to allocate 1 graph for the forward pass, 1 for the backward pass.
    params = {
      ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + 2 * ggml_graph_overhead(),
      nullptr,
      true
    };
    // This time, we use the *compute* context in order to construct the computational graph.
    // Note that after we get `_result`, we use ggml_set_output to mark _result as the output tensor.
    _ctx_compute = ggml_init(params);
    struct ggml_tensor *ax = ggml_mul(_ctx_compute, _a, _x);
    _result = ggml_add(_ctx_compute, ax, _b);
    ggml_set_output(_result);
    // To do training, we need a backend scheduler. The backend scheduler allows us to manage several backends 
    // at once for inference and training. In this case, we really only need it to fit the model. We push the CPU
    // backend as well since it is required as a fallback.
    std::vector<ggml_backend_t> backends;
    backends.push_back(_backend);
    backends.push_back(ggml_backend_cpu_init());
    _backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    std::cout << "Using " << ggml_backend_name(_backend) << " as backend\n";
    // After constructing the computational graph, we need to allocate the graph.
    // ggml_gallocr_new needs to know the backend buffer type. In this case, we 
    // find the backend buffer type using ggml_backend_get_default_buffer_type.
    _gf = ggml_new_graph(_ctx_compute);
    ggml_build_forward_expand(_gf, _result);
    _allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(_backend));
    ggml_gallocr_alloc_graph(_allocr, _gf); 
  }
```
There is a lot more going on here, but we'll go through it slowly. Note that this class takes a template parameter `T`, which allows us to specify whether we want to use `float` or `double` for the input or output values. This time around, we are going to create a *static* context and a *compute* context. The static context will be used to store tensor metadata and allocate them on the backend (the GPU). The compute context will be used to construct the computational graph and then perform the computation.

* **Lines 4-8**: The only difference from `TutorialRegression` is that we specify `no_alloc` to be `true`. This is because we will explicitly allocate the metadata for the tensors using the backend. 
* **Lines 10-16**: Since this class takes a template parameter, we determine the type of tensor (32-bit or 64-bit floating point) before proceeding to initialize the static context.
* **Lines 19-22**: We initialize the static context and then create the three tensors `_a`, `_b`, `_x` as before.
* **Lines 25-27**: Since we will be training this model, we need to specify the inputs, parameters, and outputs. This is different from `TutorialRegression`. In this case, we set `_x` as the input and `_a` and `_b` as the input. We will set `_result` to be the output later.
* **Lines 31-32**: We initialize the CUDA backend and -- since we set `no_alloc` to `true`, we now explicitly allocate the tensor metadata we created with the static context using the GPU backend.
* **Lines 39-43**: Now we create the compute context. We set `no_alloc` to `true` as before, since we will explicitly allocate the computational graph.
* **Lines 46-49**: Using the `params` we created, we initialize the compute context and then construct the computational graph. 
* **Lines 53-57**: To do training, we will need to create a backend scheduler. Note that we need to include the CPU backend in the list of backends we supply to `ggml_backend_sched_new` as a fallback.
* **Lines 61-64**: Since we specified `no_alloc = true`, we need to explicitly allocate the graph using the compute context. Computational graphs are meant to be allocated only once in GGML, but they can be used multiple times for computation once they are allocated.

Now, we [look](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/src/tutorial.cpp#L30C1-L36C2l)  at `set_params`. This time, we can't use `ggml_set_f32` since we are using a backend. Instead, we use `ggml_backend_tensor_set`:
```cpp
template<typename T>
void BackendRegression<T>::set_params(const T a, const T b) {
  // Similar to set_params for TutorialRegression. But now, we are using a backend.
  // We have to use ggml_backend_tensor_set since ggml_set_f32 is used specifically for the CPU backend. 
  ggml_backend_tensor_set(_a, &a, 0, ggml_nbytes(_a));
  ggml_backend_tensor_set(_b, &b, 0, ggml_nbytes(_b));
}
```
Let's see how `forward` is implemented. It is essentially the same as before, but since we have already pre-allocated the graph, we don't need to call `ggml_build_forward_expand`. [Here](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/src/tutorial.cpp#L38C1-L52C2)  is the code:
```cpp
template <typename T>
T BackendRegression<T>::forward(const T x) {
  // Again, we use ggml_backend_tensor_set instead of ggml_set_f32
  ggml_backend_tensor_set(_x, &x, 0, ggml_nbytes(_x));
  // We already built and allocated the graph in the constructor. 
  // Now we just have to do the computation using the backend.
  ggml_backend_graph_compute(_backend, _gf);
  // The result is stored in _result, but this shows an alternate way to get it.
  // We know that the last node in the graph is the result, we we fetch the result node.
  struct ggml_tensor *result = ggml_graph_node(_gf, -1);
  // Now, we use the backend to get the data.
  T result_data = 0;
  ggml_backend_tensor_get(result, &result_data, 0, ggml_nbytes(result));
  return result_data;
}
```
That's all there is for inference! If you read the output, it should say`Backend result: 19`. 

### Training

[Here](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/src/tutorial.cpp#L93C3-L124C4)  is the training portion for `main`:
```cpp
  /* Create 10000 datapoints; first column is x, second column is y. This is our "dataset" */
  const int N = 10000;
  float matrix[N][2];
  // Randomly generate the parameters a and b.
  std::memset(matrix, 0, 2 * N * sizeof(float));
  std::uniform_real_distribution<float> unif(1, 10);
  std::default_random_engine re;
  double a = unif(re);
  double b = unif(re);
  std::cout << "Parameters to recover: a=" << a << "; b=" << b << "\n";
  // Compute a * x + b for integer x in the interval [1, N].
  for (int i = 0; i < N; i++) {
    matrix[i][0] = static_cast<float>(i+1);
    matrix[i][1] = a * matrix[i][0] + b;
  }
  // Use the DataLoader on the matrix to create a GGML dataset.
  DataLoader<float> dl(matrix, N);
  // Train the backend regressor on the dataset.
  backend_regressor.train(dl);
  // Print the results, and evaluate at the points x = 15000, x = 20000, x = 30000
  std::cout << "Recovered parameters\n---------------\n";
  backend_regressor.print_params();
  std::cout << "Evaluation on test data\n------------\n";
  float test_x[] = { 15000.0f, 20000.0f, 30000.0f };
  for (int i = 0; i < sizeof(test_x) / sizeof(float); i++) {
    auto x = test_x[i];
    float y = a * x + b;
    float y_pred = backend_regressor.forward(x);
    std::cout << "x = " << x << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "y pred: " << y_pred << "\n";
  }
```

In particular, let's look at the `train` [method](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/src/tutorial.cpp#L54C1-L70C2):
```cpp
template<typename T>
void BackendRegression<T>::train(const DataLoader<T> &dl) {
  /* We train the model on the dataset. Here are the parameters:
   * backend_sched: The backend scheduler (we made this in the constructor) 
   * ctx_compute: The compute context (we made this in the constructor, too)
   * inputs: In this case, it is the tensor, `_x`. 
   * outputs: In this case, it is the tensor, `_result`. 
   * dataset: A ggml_opt_dataset_t. We made this inside of the DataLoader `dl`.
   * loss_type: A loss type (in this case, we use mean squared error). 
   * get_opt_pars: A callback that returns the ggml_opt_get_optimizer_params. In this case, we use 
   * the default parameters.
   * nepoch: The number of epochs.
   * nbatch_logical: How many values per batch.
   * val_split: What percentage of the data should we use for validation? We don't really need this in this example. 
   * silent: do not print diagnostic output to stderr */
   ggml_opt_fit(_backend_sched, _ctx_compute, _x, _result, dl.get_dataset(), GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, ggml_opt_get_default_optimizer_params, 5, 1, 0.2f, true);
}
```
As you can see, it is only one line -- a call to `ggml_opt_fit`. We pass the `_backend_sched` that we initialized in the constructor. We also pass in a `DataLoader` object, which returns a `ggml_opt_dataset_t` using the method `get_dataset`. A majority of the work goes into making the `ggml_opt_dataset_t`. [Here](https://github.com/svaniksharma/ggml-tutorial/blob/ca4e326a98790e3a4849e255cb89adf8e84c4db7/include/tutorial.h#L50C3-L77C4)  is the constructor of `DataLoader`:
```cpp
  DataLoader(const T matrix[][2], const size_t N) {
    // This can work for either double or float.
    enum ggml_type dataset_type;
    if (std::is_same<T, float>::value)
      dataset_type = GGML_TYPE_F32;
    else if (std::is_same<T, double>::value)
      dataset_type = GGML_TYPE_F64;
    else
      GGML_ASSERT(false);
    /* This creates a dataset. The arguments are as follows:
     * type_data: We set this above: 32-bit float or 64-bit float.
     * type_label: Same as above. In this case, it's the same type.
     * ne_datapoint: Number of elements per datapoint (i.e, how many features per datapoint) 
     * ne_label: Number of elements per label (i.e, how many outputs/targets/dependent variables do you have)
     * ndata: The number of datapoints and labels
     * ndata_shard: A shard is the unit along which a datapoint is shuffled. This is the number of points per shard. */
    _dataset = ggml_opt_dataset_init(dataset_type, dataset_type, 1, 1, N, 1);
    // Once the dataset is created, the underlying tensors are allocated for you based on the arguments passed above. ^^^
    // The following code gets the underlying tensors, and then uses ggml_get_data to get the actual buffer inside of the tensor. We then use the matrix passed in the constructor (first column datapoint, second column labels) to set these underlying buffers.
    struct ggml_tensor *data = ggml_opt_dataset_data(_dataset);
    struct ggml_tensor *labels = ggml_opt_dataset_labels(_dataset);
    T *data_buf = static_cast<T *>(ggml_get_data(data));
    T *labels_buf = static_cast<T *>(ggml_get_data(labels));
    for (int i = 0; i < N; i++) {
      data_buf[i] = matrix[i][0];
      labels_buf[i] = matrix[i][1];
    }
  }
```
Lines 3-9 determine whether we are using `float` or `double` for the dataset. Then, line 17 creates the dataset using `ggml_opt_dataset_init` (note that in this case, the output and input have the same type, though they could also be of different types). Lines 20-27 populate the `_dataset` with actual data. 

And that's it for training! The output for the training section should look something like this: 
```
Parameters to recover: a=1.00007; b=2.18384
<some diagonistic output...>
Recovered parameters
---------------
a: 1.00007
b: 2.19581
Evaluation on test data
------------
x = 15000
y: 15003.2
y pred: 15003.2
x = 20000
y: 20003.6
y pred: 20003.5
x = 30000
y: 30004.3
y pred: 30004.2

```
Note that $a$ and $b$ are generated randomly, so the numbers themselves might be different. But, if training goes correctly, the `Parameters to recover` and the `Recovered parameters` should be close in value. The predicted results should also closely match the actual `y` values, too.
