# Chapter 7. LazyAdam, or some hacks for training speed
LazyAdam proved out to be so important, that it deserves its own chapter!
## What is Adam and why is it slow?
Adam is an optimizer that keeps track of exponentially decaying 1- and 2-order moments of all the gradients. These moments (mean and variance) are used for updating the parameters on each step, instead of just the pure "gradient" computed at current step in classical Gradient Descent. This feature is supposed to help in batch-training setup which is the case for pretty much all model tranining in the world - when the model only sees a small part of the data on each step, and tries to approximate the global loss function.  

The problem with that is if you have a large embedding table, say, as we did for users - 25 million embeddings, only one embedding participating in prediction for one input record, naive Adam implementation will still update those exponentially decaying moments for ALL embeddings in the matrix. This makes it thousands of times slower than some other algorithms (like adagrad), and dependant on embedding size.  

This problem is partially mitigated by enormous batch sizes. In previous chapters, we had a batch size of 500k. This helps the model to "touch" more embeddings during one batch, partially mitigating the issue of updating momentums for untouched embeddings. In practice, increasing batch size with this naive Adam implementation helps speed up the training, especially on TPU, but might introduce other problems, for example with memory or convergence.

## Introducing: lazy adam!
Here is the relevant discussion: https://groups.google.com/a/tensorflow.org/g/discuss/c/6GvpMz8kb-U/m/FaBAJkbvEQAJ  

LazyAdam is an alternative implementation of Adam that updates momentums only for embeddings actually touched in current batch. It is available in two flavors: as an optimizer in tensorflow_addons package: [LazyAdam](https://www.tensorflow.org/addons/tutorials/optimizers_lazyadam), and as an option for TPU optimizer [tf.tpu.experimental.embedding.Adam](https://www.tensorflow.org/api_docs/python/tf/tpu/experimental/embedding/Adam) in later versions of TF (>= 2.11.0). In my case, speed up was up to 10x on cpu (or 1000x on smaller batches), and about 2-2.5x on TPU.