### 1. Sparse-Optimized Adam.
As most of gradient of embedding layer is zero, we can avoid the useless computing of zeros. This implement equal to Adam mathematically.

### 2. Parallel model computing.
We leverage BigDL to do a data parallelism optimization, we split the minibatch into `n` pieces, can create `n` model, each model run a pieces of minibatch.

### 3. Parallel DataGeneration
Using multiThread and multi RNG to generate data.

### 4. Parallel data shuffle
As shuffle cost a lot of time, we use multithread to shuffle the data. We shuffle two times of all the data, with different kinds of bucket. 
