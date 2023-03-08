# Chapter 5. Tidying up
In the last chapter, we've written a collaborative filtering models with embedding lookup layer and dot product. We tried training it on cpu with a little bit of data.  
Before we scale that up to big datasets, tpus, and do some serious experimenting, we have some stuff in our code that we need to clear out:
- We have all our code in one file - jupyter notebook. We want to split it into several modules - model code, dataset code; a proper training script that can be run from command line; helper jupyter notebook that could access dataset and model modules - this is useful for development, when you want to actually load a bit of data, instantiate the model, look into inputs/outputs and try new stuff out interactively
- We have a lot of constants in our code. We want to move it to command-line arguments and/or config. This include training/input pipeline options (paths, iterations, batch_size etc), input feature disposition (it is likely that at some point in the future we will be training models with different fields), model configuration (more layers, embedding sizes, vocabulary sizes)
- We need to save the model, as well as some intermediate metrics, learning history, run parameters and stuff like that
Let s go ahead and fix them!
## Code configuration: config and command line
There is no one answer to which one is better - config or command line. Most of my career I've used a c++ tool that parsed all input options for command line. Command line arguments make it easy to modify runs and experiments; configs allow more structured, nested definitions; configs are usually commited to Github, which helps make experiments more reproducible.  
Here, we will be using configs. I've opted for the `yaml` format, it is quite clean and easy to read and write. This is how it is supposed to work:
```
%%writefile test_config.yaml
batch_size: 1024
dataset_features:
    label:
        type: "int"
    userId:
        type: "int"
    movieId:
        type: "float"
    timestamp:
        type: "int"
```
```
import yaml
config = yaml.safe_load(open("test_config.yaml", "r"))
config
```
```
{'batch_size': 1024,
 'dataset_features': {'label': {'type': 'int'},
  'userId': {'type': 'int'},
  'movieId': {'type': 'float'},
  'timestamp': {'type': 'int'}}}

```
`yaml` lib loads an arbitrary yaml file into a nested python structure with dicts/lists.  
Here is how the full config shoud look like:
```
%%writefile config.yaml
epochs: 2
global_batch_size: 8192
shuffle_buffer: 10
train_rows: 65536
trainval_rows: 8192
eval_rows: 65536
compression: "GZIP"
train_path: "gs://mgaiduk-us-central1/ratings/csv_gzip/part*"
validate_path: "gs://mgaiduk-us-central1/ratings_validate/csv_gzip/part*"
save_model_path: "gs://mgaiduk-us-central1/models/model1"
cycle_length: 8
dataset_features:
    userId:
        type: "string"
    movieId:
        type: "string"
    label:
        type: "int"
    timestamp:
        type: "int"
label: label
model:
    learning_rate: 0.01
    features:
        userId:
            hash: true
            vocab_size: 25000000
            embedding_dim: 16
            belongs_to: user
        movieId:
            hash: true
            vocab_size: 5000000
            embedding_dim: 16
            belongs_to: movie
```
Note that here, we describe the features twice: once to properly parse the input, once - to specify the model. This sounds redundant, but might be handy when you are running different models on the same dataset, and not all of the models use all of the features.  
We might have more features in the future (like user country, movie genre, etc), but the final layer of the model will still be a dot product between two embeddings. Extra features will have to belong either to the user part of the model, or to the movie part. This architecture is called "late binding" and is useful for future use at runtime: there are data structures that allow doing fast search for K Nearest Neighbors (KNN) in such a case. Examples are HNSW, SCANN. Typically, movie embeddings are baked into the index, while user embeddings are fetched on-the-fly.

I'd rather use some proper class fields than dictionary keys, so I wrote a parser for this config:
```
%%writefile config.py
import yaml

class DatasetFeature:
    def __init__(self, feature_name, dic):
        self.name = feature_name
        self.type = dic["type"]
    
    def __repr__(self):
        return "DatasetFeature: " + str(self.__dict__)

class Feature:
    def __init__(self, feature_name, dic):
        self.hash = False
        if "hash" in dic:
            self.hash = dic["hash"]
            if self.hash:
                assert "vocab_size" in dic
                self.vocab_size = dic["vocab_size"]
        self.embedding_dim = dic["embedding_dim"]
        self.name = feature_name
        self.belongs_to = dic["belongs_to"]

    def __repr__(self):
        return "Feature: " + str(self.__dict__)
        
class Model:
    def __init__(self, dic):
        self.learning_rate = dic["learning_rate"]
        self.features = []
        for feature_name, feature_dic in dic["features"].items():
            self.features.append(Feature(feature_name, feature_dic))

    def __repr__(self):
        return "Model: " + str(self.__dict__)

class Config:
    def __init__(self, path):
        dic = yaml.safe_load(open(path, 'r'))
        self.epochs = dic["epochs"]
        self.compression = dic["compression"]
        self.global_batch_size = dic["global_batch_size"]
        self.label = dic["label"]
        self.shuffle_buffer = dic["shuffle_buffer"]
        self.train_path = dic["train_path"]
        self.validate_path = dic["validate_path"]
        self.save_model_path = dic["save_model_path"]
        self.train_rows = dic["train_rows"]
        self.trainval_rows = dic["trainval_rows"]
        self.eval_rows = dic["eval_rows"]
        self.model = Model(dic["model"])
        self.dataset_features = []
        self.cycle_length = dic["cycle_length"]
        for feature_name, feature_dic in dic["dataset_features"].items():
            self.dataset_features.append(DatasetFeature(feature_name, feature_dic))
       
    def __repr__(self):
        return "Config: " + str(self.__dict__)
```
Nothing special is happening here. Here is how we can use this code:  
```
from config import Config
config = Config("config.yaml")
config
```

We move dataset parsing code to a separate file:
```
%%writefile dataset.py
import tensorflow as tf

class DatasetReader:
    def __init__(self, config, path):
        self.config = config
        self.path = path
        defaults = []
        for feature in self.config.dataset_features:
            if feature.type  == "int":
                defaults.append(tf.constant(0, dtype=tf.int64))
            elif feature.type == "float":
                defaults.append(tf.constant(0.0, dtype=tf.float32))
            elif feature.type == "string":
                defaults.append(tf.constant("", dtype=tf.string))
            else:
                assert False
        self.defaults = defaults

    def __call__(self, ctx: tf.distribute.InputContext):
        batch_size = ctx.get_per_replica_batch_size(
            self.config.global_batch_size) if ctx else self.config.global_batch_size
        @tf.function
        def decode_fn(record_bytes):
            csv_row = tf.io.decode_csv(record_bytes, self.defaults)
            parsed_features = {}
            for i, feature in enumerate(self.config.dataset_features):
                parsed_features[feature.name] = csv_row[i]
            features = {}
            for feature in self.config.model.features:
                t = parsed_features[feature.name]
                features[feature.name] = t
            labels = {
                "label": parsed_features[self.config.label]
            }
            return (features, labels)

        def make_dataset_fn(path):
            dataset = tf.data.TextLineDataset([path], compression_type=self.config.compression.upper())
            dataset = dataset\
                .shuffle(self.config.shuffle_buffer)\
                .batch(batch_size, drop_remainder=True)\
                .repeat(self.config.epochs).map(decode_fn)
            return dataset
        filenames = tf.data.Dataset.list_files(self.path, shuffle=True, seed=42)
        if ctx and ctx.num_input_pipelines > 1:
            filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        dataset = filenames.interleave(make_dataset_fn, num_parallel_calls=10, deterministic=False, cycle_length=10)
        dataset = dataset.prefetch(100)
        return dataset

def create_dataset(config, strategy, path):
    dataset_callable = DatasetReader(
        config=config,
        path=path
    )
    dataset = strategy.distribute_datasets_from_function(
        dataset_fn=dataset_callable,
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False),
    )
    return dataset
```
We rewrote the code to use parameters from config instead of hard-coded ones, which includes expected defaults for csv parser.  

We also turned our dataset code into a callable struct - this is a requirement to be able to use it in a distributed strategy.  
This callable is expecting to be called with the following signature:  
`def __call__(self, ctx: tf.distribute.InputContext):`  
`ctx` will be passed to the call, in case we launch the training in a distributed context. If we do not, there just will be a `None`, but our code should work still.

`batch_size = ctx.get_per_replica_batch_size(...)`: in multihost TPU or GPU training under data parallelism, the distribution disposition will look like this: there will be N workers, each reading its own portion of the dataset. One worker reads from the dataset in batches with size `batch_size`, and gives it to one of its K  devices (TPU cores or GPU devices). The devices then store that batch in the on-device memory, process it independently, but accumulate gradients in sync. Thus, the effective batch size (which might affect training quality, not just speed) will be batch_size * K = global_batch_size. Conveniently, the context provides the function to calculate batch size from global_batch_size, since the information about total device count available to this worker is stored in this context.  

```
if ctx and ctx.num_input_pipelines > 1:
    filenames = filenames.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
```
Here we do the sharding, again, knowing the total number of workers and current worker id.  

`dataset = strategy.distribute_datasets_from_function(...)` - this call is needed to announce that instead of reading the dataset locally, on the machine that is launching the training, we will be executing this code remotely on multiple workers.
`options=tf.distribute.InputOptions(experimental_fetch_to_device=False)`: this fixes the error `ValueError: Received input tensor postId which is on a TPU input device /job:worker/replica:0/task:0/device:TPU:0. Input tensors for TPU embeddings must be placed on the CPU. Please ensure that your dataset is prefetching tensors to the host by setting the 'experimental_prefetch_to_device' option of the dataset distribution function. See the documentation of the enqueue method for an example.`. I dont quite know what it means.  

Let's test this dataset code:  
```
import tensorflow as tf
from dataset import create_dataset
strategy = tf.distribute.get_strategy()
dataset = create_dataset(config, strategy, config.train_path)
for elem in dataset:
    break
elem
```
Our code is getting cleaner! Creating a dataset now takes just 2 lines of code, provided that we use the module that we created earlier. And we can still use it with a local strategy, iterate over the elements, print them out, try to apply some layers to them.  

## The model
```
%%writefile model.py
import math
import tensorflow as tf
import tensorflow_recommenders as tfrs

class BaseModel(tfrs.models.Model):
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
    
    def call(self, inputs):
        raise NotImplementedError

    def compute_loss(self, inputs, training=False):
        features, labels = inputs
        outputs = self(features, training=training)
        # loss = tf.reduce_mean(label_loss)
        loss = self.task(labels=labels["label"], predictions=outputs["label"])
        loss = tf.reduce_mean(loss)
        return loss

class Model(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = config.model.learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = config.model.learning_rate)
        self.hashing_layers = {}
        embedding_layer_feature_config = {}
        for feature in self.config.model.features:
            if feature.hash:
                self.hashing_layers[feature.name] = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=feature.vocab_size)
            initializer = tf.initializers.TruncatedNormal(
                mean=0.0, stddev=1 / math.sqrt(feature.embedding_dim)
            )
            embedding_layer_feature_config[feature.name] = tf.tpu.experimental.embedding.FeatureConfig(
                table=tf.tpu.experimental.embedding.TableConfig(
                vocabulary_size=feature.vocab_size,
                initializer=initializer,
                dim=feature.embedding_dim))
        self.embedding_layer = tfrs.layers.embedding.TPUEmbedding(
            feature_config=embedding_layer_feature_config,
            optimizer=self.embedding_optimizer)
        self.final_activation = tf.keras.layers.Activation('sigmoid')
        

    def call(self, inputs):
        features = {}
        for feature in self.config.model.features:
            t = inputs[feature.name]
            if feature.hash:
                t = self.hashing_layers[feature.name](t)
            features[feature.name] = t
        embeddings = self.embedding_layer(features)
        user_embs = []
        movie_embs = []
        for feature in self.config.model.features:
            embedding = embeddings[feature.name]
            if feature.belongs_to == "user":
                user_embs.append(embedding)
            elif feature.belongs_to == "movie":
                movie_embs.append(embedding)
            else:
                assert False
        user_final = tf.concat(user_embs, axis = 1)
        movie_final = tf.concat(movie_embs, axis = 1)
        # last unit of embedding is considered to be bias
        # out = tf.keras.backend.batch_dot(user_final[:, :-1], post_final[:, :-1]) + user_final[:, -1:] +  post_final[:, -1:]
        # This tf.slice code helps get read of "WARNING:tensorflow:AutoGraph could not transform ..." warnings produced by the above line
        # doesn't seem to improve speed though
        # user_final_emb = tf.slice(user_final, begin=[0, 0], size=[user_final.shape[0],  user_final.shape[1] - 1])
        # user_final_bias = tf.slice(user_final, begin=[0, user_final.shape[1] - 1], size=[user_final.shape[0],  1])
        # movie_final_emb = tf.slice(movie_final, begin=[0, 0], size=[movie_final.shape[0],  movie_final.shape[1] - 1])
        # movie_final_bias = tf.slice(movie_final, begin=[0, movie_final.shape[1] - 1], size=[movie_final.shape[0],  1])
        user_final_emb = user_final[:,:-1]
        user_final_bias = user_final[:,-1:]
        movie_final_emb = movie_final[:,:-1]
        movie_final_bias = movie_final[:,-1:]
        out = tf.keras.backend.batch_dot(user_final_emb, movie_final_emb) + user_final_bias + movie_final_bias
        prediction = self.final_activation(out) 
        return {
            "label": prediction
        }
```
Nothing particularly interesting happening here - we just rewrote the code to use the config for parameters/feature names and stuff like that.  

Here is how we can construct the model now:
```
from model import Model
with strategy.scope():
    model = Model(config)
    model.compile(model.optimizer, steps_per_execution=10)
model(elem[0]) # see some model predictions
```
## The traning loop

```
import sys
import os
def save_string_gcs(string_object, gcs_dir, filename):
    string_string = json.dumps(string_object)
    with open(filename, "w") as f:
        f.write(string_string)
    os.system(f"gsutil -m cp {filename} {gcs_dir}/{filename}")
    os.system(f"rm {filename}")

train_dataset = create_dataset(config, strategy, config.train_path)
trainval_dataset = create_dataset(config, strategy, config.validate_path)
eval_dataset = create_dataset(config, strategy, config.validate_path)
train_steps_per_epoch = config.train_rows // config.global_batch_size
trainval_steps_per_epoch = config.trainval_rows // config.global_batch_size
eval_steps_per_epoch = config.eval_rows // config.global_batch_size
checkpoints_cb = tf.keras.callbacks.ModelCheckpoint(config.save_model_path  + '/checkpoints/',  save_freq = train_steps_per_epoch//3)
callbacks=[checkpoints_cb]
history = model.fit(train_dataset, epochs=config.epochs, callbacks=callbacks, steps_per_epoch=train_steps_per_epoch,
validation_data=trainval_dataset, validation_steps=trainval_steps_per_epoch)
model.save_weights(config.save_model_path  + '/weights/')
eval_steps = config.eval_rows // config.global_batch_size
eval_scores = model.evaluate(eval_dataset, return_dict=True, steps=eval_steps_per_epoch)
metrics = {}
metrics["eval"] = eval_scores
metrics["history"] = history.history
metrics["args"] = sys.argv
metrics["config"] = repr(config)
save_string_gcs(json.dumps(metrics), config.save_model_path, f"metrics_pretrain.json")
```
There are some interesting things here.  
`train_steps_per_epoch = config.train_rows // config.global_batch_size`: as is typical for neural network training, training is split into several epochs. Training loop does some extra evaluations at the end of every epochs. In our dataset code, we assume that every epoch reads the entire dataset (hence `dataset.repeat(epochs)`), but that is not necessary. Important thing is that our dataset is limited in size, and that size is not known to the training loop. We have to provide it externally and to do it correcly, otherwise training will fail with "dataset exhausted" error. Steps are measured in batches, so we have to know our original data size (can check it in Bigquery) and divide it by batch size.  

`checkpoints_cb = tf.keras.callbacks.ModelCheckpoint` - this is the callback that will save model weights during the training. We can provide `gs://` path as directory to save model to.  

`model.save_weights()` - save final model after the tranining  

`eval_scores = model.evaluate()` - do final evaluation on full eval dataset (evaluation on part of the dataset was done during training)  

`save_string_gcs()` - we save training history, final evaluation results, config and run arguments to gcs as well for reproducibility.  

## Conclusion
By now, we have a complete, neat, configurable code to train our model on CPU on one worker. Next, we will discuss how to launch tranining on TPUs.  

As usual, code for this chapter is available as [jupyter notebook](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter5/tidying_up.ipynb). You should also checkout the [jupyter notebook playground](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter5/playground.ipynb) that allows you to load dataset and models from the module files that we wrote, and [train script](https://github.com/mgaiduk/mgaiduk.github.io/blob/main/my-first-book/src/code/chapter5/train.py) that you can launch from command line to train the model.