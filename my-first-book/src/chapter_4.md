# Chapter 4. The model
Now that we have our data pipeline, let's build the model. It will be a classical collaborative filtering model, with a layer looking up unique userId's and movieId's into N-dim embeddings, then dot-product, activation and that's it. To do that, we first need to map unique ids to continuous integers from 0 to vocab_size.  

There are 2 main ways to do that. One way is to collect a "dictionary", mapping tokens to integers based on their frequency, and mapping all tokens that are rare enough into a special "out of vocabulary" OOV token. This can be done in Tensorflow directly, though a more scalable solution usually is to do it as a preprocessing step in Bigquery/MapReduce.  
The second way is to take hash(id) % vocab_size. Hash itself, as well as modulo operator, will inevitably lead to collisions, which cross-contaminates the signal between different entities and possibly reducing the quality. Hashing is, however, much easier to set up, especially in an "incremental training" setting. We will do some benchmarks later to determine which way is better in respect to quality.  

First, here is the code from previous chapter to read the dataset:
```
import tensorflow as tf
print(tf.__version__)
import tensorflow_recommenders as tfrs
print(tfrs.__version__)

import math
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
dataset = filenames.interleave(make_dataset_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False, cycle_length=8)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
for elem in dataset:
    break
elem
```
Here is how we can hash the input:
```
vocab_size = 1000
hashing_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=vocab_size)
hashing_layer(elem[0]["userId"]) # it's good that we have an eager tensor on our hands, so we can test inputs-outputs to all parts of our model, heh?
```
Having hashing layer instead of, say, hashing preprocessing step is good because we can have vocab_size information baked into our model. Different models can have different vocab_size, at which point it becomes necessary to track that information and synchronize it on the preprocessing step of the inference side, which is usually done in a language other than python and tensorflow. Having hashing as just an another layer is a handy way to avoid that.  
`vocab_size` is a hyper-parameter that, ideally, needs to be tuned. Ideally, it should be about the same size as the number of unique IDs. However, entity id's histogram is usually very skewed: with just a few most popular ids getting most of the attention, and "long tail" of rare ids (about 50% of total mass, typically) having just one or a few interactions associated with them. In our CF model, embedding tables will be the most expensive part. 25 millions of unique entities converted to embeddings of 32 4-byte floating point numbers is already 3 gb of data, that will need to be stored in-memory during training and inference time. So this parameter deserves some experimenting to better approximate the trade-offs.  

Now we need an embedding layer. There are 2 TF layers that do that that I am aware of: [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) and [tfrs.layers.embedding.TPUEmbedding](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/embedding/TPUEmbedding). First one can be used on CPU and GPU. Second one can be used on CPU and TPU. APIs are different, unfortunately. I trained my models on cpu and tpu, so we are sticking with TPUEmbedding. Luckily, cpu quality and performance of both of these layers is identical. Here is how to use it:
```
embedding_dim = 8
lr = 0.01
initializer = tf.initializers.TruncatedNormal(
    mean=0.0, stddev=1 / math.sqrt(embedding_dim)
)
embedding_layer_feature_config = {
    "userId": tf.tpu.experimental.embedding.FeatureConfig(
        table=tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        initializer=initializer,
        dim=embedding_dim)),
    "movieId": tf.tpu.experimental.embedding.FeatureConfig(
        table=tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        initializer=initializer,
        dim=embedding_dim)),
}
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = lr)
embedding_layer = tfrs.layers.embedding.TPUEmbedding(
    feature_config=embedding_layer_feature_config,
    optimizer=optimizer)
hashed_tensor = {
    "userId": hashing_layer(elem[0]["userId"]),
    "movieId": hashing_layer(elem[0]["movieId"])
}
embeddings = embedding_layer(hashed_tensor)
embeddings
```
Note here that we only need one embedding layer for all our inputs. It gets a dictionary {name -> tensor} as input, and provides one as an output. Output tensors have shape (batch_size, embedding_dim). Also note that vocab size in the embedding table is the same as we used for hashing layer.   
Now we just need to do dot product. The API looks as follows:
```
tf.keras.backend.batch_dot(user_emb, movie_emb)
```
If user_emb and movie_emb are vectors of the shape (batch_size, emb_dim), `batch_dot` computes per-element dot product.  
We also want to have "bias" for users and movies. For example, for imdb movies, "8.7 out of 10" rating can be represented as a movie bias - unpersonalized, overall level of popularity of the movie. The part of the embedding that goes into dot product - personalozed signal, that has no meaning without corresponding user embedding. I found out the hard way that without bias, the model will hardly be able to train, so this is a crucial peace of information. Here is the code to compute dot product + bias:
```
user_emb = embeddings["userId"]
movie_emb = embeddings["movieId"]
user_final_emb = tf.slice(user_emb, begin=[0, 0], size=[user_emb.shape[0],  user_emb.shape[1] - 1])
user_final_bias = tf.slice(user_emb, begin=[0, user_emb.shape[1] - 1], size=[user_emb.shape[0],  1])
movie_final_emb = tf.slice(movie_emb, begin=[0, 0], size=[movie_emb.shape[0],  movie_emb.shape[1] - 1])
movie_final_bias = tf.slice(movie_emb, begin=[0, movie_emb.shape[1] - 1], size=[movie_emb.shape[0],  1])
out = tf.keras.backend.batch_dot(user_final_emb, movie_final_emb) + user_final_bias + movie_final_bias
out
```
We say that last element of the embedding is treated as bias (since it is just another user/movie dependant number, same as embedding vector components), while all except last - go into dot product. This is represented with `tf.slice` function because, for some reason, normal python indexing produced errors during TF graph compilation for me.  

All our crucial pieces are in place, time to build the final model! We will build a subclass style model:
```
class Model(tfrs.models.Model):
    def __init__(self, ...)
        ...
    def call(self, inputs):
        ...
    def compute_loss(self, inputs, training=False):
        ...
```
If we put all our layers code in a model, we get a few benefits, such as:
- The ability to call model.fit(dataset) method instead of writing a custom training loop with ineffective for loop in Python
- Ability to save and load model weights
- Ability to use "task" API that lets you easily add trackable metrics to your training, such as avg loss, AUC, prediction distribution etc  
For a subclass-style model, we have to:
1. Inherit from one of [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) or [tfrs.models.Model](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/models/Model). Second one is from `tensorflow_recommenders` which is recommended for users of TPU Embedding Layer.
2. Define a `call` method. When training, input pipeline is supposed to output tuples of (features, targets) objects. `call()` method will get the first element of this tuple, i.e., features. The object can be anything, but here it is a dictionary of {feature_name: tensor}. This method is used to get predictions for the model from the given input data
3. Define a `compute_loss` method. Unlike `call`, this method gets a tuple of (features, targets) as an input. Failure to realize that will cause your pipeline to fail with some weird errors, like "no gradient was provided for ...". `compute_loss` should output a scalar loss that will be used in minimization and gradient computing.
Here is how the final code will look for us:
```
class Model(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            metrics=[
                tf.keras.metrics.BinaryCrossentropy(name="label-crossentropy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr-auc"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            ],
            prediction_metrics=[
                tf.keras.metrics.Mean("prediction_mean"),
            ],
            label_metrics=[
                tf.keras.metrics.Mean("label_mean")
            ]
        )
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = lr)
        self.hashing_layer = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=vocab_size)
        embedding_layer_feature_config = {
            "userId": tf.tpu.experimental.embedding.FeatureConfig(
                table=tf.tpu.experimental.embedding.TableConfig(
                vocabulary_size=vocab_size,
                initializer=initializer,
                dim=embedding_dim)),
            "movieId": tf.tpu.experimental.embedding.FeatureConfig(
                table=tf.tpu.experimental.embedding.TableConfig(
                vocabulary_size=vocab_size,
                initializer=initializer,
                dim=embedding_dim)),
        }
        self.embedding_layer = tfrs.layers.embedding.TPUEmbedding(
            feature_config=embedding_layer_feature_config,
            optimizer=self.optimizer)
        self.final_activation = tf.keras.layers.Activation('sigmoid')
        

    def call(self, inputs):
        hashed_inputs = {}
        for field in ["userId", "movieId"]:
            hashed_inputs[field] = self.hashing_layer(inputs[field])
        print("Hashed inputs: ", hashed_inputs)
        embeddings = self.embedding_layer(hashed_inputs)
        user_emb = embeddings["userId"]
        movie_emb = embeddings["movieId"]
        # last unit of embedding is considered to be bias
        # out = tf.keras.backend.batch_dot(user_final[:, :-1], post_final[:, :-1]) + user_final[:, -1:] +  post_final[:, -1:]
        # This tf.slice code helps get read of "WARNING:tensorflow:AutoGraph could not transform ..." warnings produced by the above line
        # doesn't seem to improve speed though
        user_final_emb = tf.slice(user_emb, begin=[0, 0], size=[user_emb.shape[0],  user_emb.shape[1] - 1])
        user_final_bias = tf.slice(user_emb, begin=[0, user_emb.shape[1] - 1], size=[user_emb.shape[0],  1])
        movie_final_emb = tf.slice(movie_emb, begin=[0, 0], size=[movie_emb.shape[0],  movie_emb.shape[1] - 1])
        movie_final_bias = tf.slice(movie_emb, begin=[0, movie_emb.shape[1] - 1], size=[movie_emb.shape[0],  1])
        out = tf.keras.backend.batch_dot(user_final_emb, movie_final_emb) + user_final_bias + movie_final_bias
        prediction = self.final_activation(out) 
        return {
            "label": prediction
        }
    def compute_loss(self, inputs, training=False):
        features, labels = inputs
        outputs = self(features, training=training)
        # loss = tf.reduce_mean(label_loss)
        loss = self.task(labels=labels["label"], predictions=outputs["label"])
        print(loss)
        loss = tf.reduce_mean(loss)
        return loss
```
New stuff here:  
- `self.task = tfrs.tasks.Ranking(...)` - this convenience field lets us define loss as well as a few extra metric to be tracked during training and evaluation  
- `self.final_activation = tf.keras.layers.Activation('sigmoid')` - we add activation function after dot product
Now to finally do some training:
```
strategy = tf.distribute.get_strategy()
with strategy.scope():
    model = Model()
    model.compile(model.optimizer, steps_per_execution=10)
    model.fit(dataset, epochs=1, steps_per_epoch=1000)
```
```
Tensor("while/ranking_3/Identity:0", shape=(), dtype=float32)
1000/1000 [==============================] - 2s 2ms/step - label-crossentropy: -0.9844 - auc: 0.0000e+00 - pr-auc: 1.0000 - accuracy: 0.0000e+00 - prediction_mean: 0.7430 - label_mean: 2.0000 - loss: -1.0230 - regularization_loss: 0.0000e+00 - total_loss: -1.0230
```
Here, we initialize a "default strategy" that is just a layer of compatibility for when we start distributed/TPU training. We compile the model, turning our python code for calling the model and computing loss into proper optimized TF graphs. `steps_per_execution` option allows you to roll out several training loop iterations into just one graph execution, which might speed it up in some weird cases.  

This should train our model for a little bit, outputting our loss as well as some extra metrics, as promised.  

This was a good start, we can already train a model, start experimenting, see some metrics. In the next chapter, we are going to tidy up the code a little bit.  

As usual, code from this chapter is available as [jupyter notebook](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter4/model_training.ipynb)