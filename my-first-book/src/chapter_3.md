# Chapter 3. Proper input pipeline and the model
In the last chapter, we discussed various data storing formats a little bit, and got to take a first look at the data, still not parsed, but already represented as a tensor.  
In this chapter, we will build a proper data pipeline, reading data from all our files, parsing it and properly handling sharding during distributed training.
### CSV data parsing
In the last chapter, we had roughly the following code:
```
dataset = tf.data.TextLineDataset(["gs://mgaiduk-us-central1/ratings/csv_gzip/part000000000000"], compression_type="GZIP")
for line in dataset.batch(16): # batching added to demonstrate that parsing works on higher dimensional tensors
    break
print(line)
```
That reads the data from just one file out of several for our exported table, and does no parsing.  

Here is how we can parse it:
```
defaults = [tf.constant(0, dtype=tf.int64),
           tf.constant(0, dtype=tf.int64),
           tf.constant(0, dtype=tf.float32),
           tf.constant(0, dtype=tf.int64)]
csv_row = tf.io.decode_csv(line, defaults)
csv_row
```
`[<tf.Tensor: shape=(16,), dtype=int64, numpy=array([  70,  188,  243,  359,  426,  440,  446,  626,  634,  700,  734,1044, 1332, 1367, 1409, 1459])>,`  
`<tf.Tensor: shape=(16,), dtype=int64, numpy=array([  3948,    653, 103249,   3578,   1500,   3022,   2861,   2162,1006,  46578,   3203,   2406,   1198,   1037,   5530,   1183])>,`  
`<tf.Tensor: shape=(16,), dtype=float32, numpy=array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],dtype=float32)>,`  
`<tf.Tensor: shape=(16,), dtype=int64, numpy=array([1255219128, 1025333400, 1464280162,  974700907, 1371502543,1231472902, 1017900083, 1002897585,  865372001, 1350795469,992217420,  944907698, 1529895670, 1185835950, 1288302165,889019866])>]`  

We pass `defaults` to it to signify which data types are expected, and to use in case an entire column is missing (though it is rather hard to imagine in CSV)  

To be able to just call `model.fit(training_dataset)` on our data, we need to turn it into a dataset, and have it output a tuple with 2 values: features and labels. It is also common to represent features as a dictionary, to know what column means what, and to be able to connect it to model code. Here is the code that does that:
```
@tf.function
def decode_fn(csv_line):
    defaults = [tf.constant(0, dtype=tf.int64),
           tf.constant(0, dtype=tf.int64),
           tf.constant(0, dtype=tf.float32),
           tf.constant(0, dtype=tf.int64)]
    csv_row = tf.io.decode_csv(csv_line, defaults)
    features = {}
    features["userId"] = csv_row[0]
    features["movieId"] = csv_row[1]
    labels = {
        "label": csv_row[2]
    }
    return (features, labels)
dataset = tf.data.TextLineDataset(["gs://mgaiduk-us-central1/ratings/csv_gzip/part000000000000"],
                                  compression_type="GZIP").batch(16, drop_remainder=True).map(decode_fn)
for elem in dataset:
    break
elem # take a look at the resulting data
```
`dataset.map()` function takes a dataset and applies a function to each row.  
We do the batching first, and then do the parsing. This is because we want to spend less time in our own, python code, and more time in optimized tensorflow intrinsics. Because of this, it is better to turn small tensors into big, high-dimensional tensors first, to have our python code benefit from vectorization.  

Another interesting detail is the works of @tf.function. It does roughly the following: upon calling the function for the first time, TF engine sees tensor inputs with concrete shapes, and compiles the code into an execution graph, treating all non-tensor inputs as contants. If non-tensor inputs are not constant, say, you have some training-dependable variable in the pipeline, graph compilation will happen every time, which will probably be slower then just running the python code. All side effects are also executed on graph compile time only. This means that you can insert print statements in the desired places in your code, and see what exactly is being passed around, but only on the first step of your execution.  

If tensor shapes change during execution, the training will probably crush (at least on TPU). This is why we have `.batch(16, drop_remainder=True). Normally, when batching limited size dataset, last batch is slightly smaller. We don't want our model to crush right at the end, so we just drop that remainder.  

Next, we need to read all the files for our dataset, not just the one file. We could do it like this:
```
filenames = tf.data.Dataset.list_files("gs://mgaiduk-us-central1/ratings/csv_gzip/part*", shuffle=True, seed=42)
dataset = tf.data.TextLineDataset(filenames,
                                  compression_type="GZIP").batch(16, drop_remainder=True).map(decode_fn)
for elem in dataset:
    break
elem
```
`TextLineDataset` api is smart enough to accept a list of files, or even a dataset that outputs filenames. The only reason NOT to do it like that is proper sharding, which should be done like this instead:
```
ctx = None # will not be none in distributed strategy, see later
def make_dataset_fn(path):
    dataset = tf.data.TextLineDataset([path], compression_type="GZIP")
    dataset = dataset\
        .batch(16, drop_remainder=True)\
    .map(decode_fn)
    return dataset
filenames = tf.data.Dataset.list_files("gs://mgaiduk-us-central1/ratings/csv_gzip/part*", shuffle=True, seed=42)
if ctx and ctx.num_input_pipelines > 1:
    filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
dataset = filenames.interleave(make_dataset_fn)
for elem in dataset:
    break
elem
```
`tf.data.Dataset.list_files` creates a dataset with all filenames matching a pattern.  
When we train under a distributed strategy, we will receive a ctx with information about total number of shards and current shard index. Under data parallelism methodology, we want each shard to read its own portion of data. It can be achieved by `tf.data.Dataset.shard()` function, that takes total number of batches and shard idx as input. It works like this: it reads the dataset and skips all the data except 1/nths. We don't want to actually read and parse all that skipped data; that is why we call this method on filenames dataset.  

Finally, we want to add some dataset stuff to make sure input pipeline is not a bottleneck:
```
ctx = None # will not be none in distributed strategy, see later
filenames = tf.data.Dataset.list_files("gs://mgaiduk-us-central1/ratings/csv_gzip/part*", shuffle=True, seed=42)
if ctx and ctx.num_input_pipelines > 1:
    filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
dataset = filenames.interleave(make_dataset_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False, cycle_length=8)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

for elem in dataset:
    break
elem
```
New things here:  

1. We call `prefetch` dataset method. Without it, one iteration of training loop will call dataset processing code to fetch 1 element, then perform a model training step, then call dataset function and so on. If you are doing your training on a multi-core machine, all those extra cores will probably just do nothing during dataset processing step. Having prefetch allows you to process dataset and do model training at the same time. AUTOTUNE parameter usually works, but if you are really interested in trying out different values - you can put some number here.  
Important to note that we do `prefetch` in the very end, making sure we prefetch fully parsed, batched data.  

1. dataset.interleave got 2 new parameters: num_parallel_calls and cycle_length. They both are required for parallelism because of how parallel computing works in `interleave` calls. This is the second crucial step in making sure your input pipeline is not a bottleneck. If your parsing is done on just one core, having prefetch will not be enough to fetch new data with enough speed. Parallel dataset execution makes sure that you give enough cores for the task  

1. `deterministic=False` in `interleave`. In theory, this might speed up input data pipeline, because it allows dataset executor to output results from multiple parallel calls as soon as they are ready. However, if you have prefetch and parallel calls, you probably won't notice latency differences. Important thing is, if you are doing distributed training, to have sharding done BEFORE any non-determinism in the pipeline, otherwise the shards will train on random, partially intersecting parts of data. 

Code for this chapter is available as a [jupyter notebook](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter3/proper_dataset_parsing.ipynb).