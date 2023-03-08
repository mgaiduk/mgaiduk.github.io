# Chapter 6. Training on TPU
## What are TPUs?
TPUs are ASICs from Google designed specifically for neural networks training. Processing cores are connected in such a way as to allow typical model training workflows, like matrix multiplication, to be done quickly without intermediate memory reads and writes. In addition, worker node was designed with typical training task in mind, and so it has A LOT OF MEMORY (which is a big problem for GPUs).   

In addition, TPUs are very expensive. They do allow you to scale up your training in some cases, like deep neural network training. In other cases, they are powerful but not cost-effective. One such case happens to be collaborative filtering - the model itself is quite shallow in such a case, training speed is bounded by input pipeline throughput rather then actual training compute. It is, however, handy to be able to quickly try them out and do the experiment yourself.  

Luckily, we wrote our code in such a way as to make it easy to apply on TPUs.  

First, do some modifications to our training code:
```
# train.py
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, help="Path to config")
parser.add_argument("--tpu_name", type=str)
parser.add_argument("--tpu_zone", type=str)
args = parser.parse_args()
config = Config(args.config)
if args.tpu_name:
    assert args.tpu_zone
    handle = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name, zone=args.tpu_zone)
    tpu = handle.connect(args.tpu_name, args.tpu_zone)
    strategy = tf.distribute.TPUStrategy(tpu)
    tf.tpu.experimental.initialize_tpu_system(handle)
    print("num_replicas_in_sync: ", strategy.num_replicas_in_sync)
else:
    strategy = tf.distribute.get_strategy()
```
We read tpu_name and tpu_zone from command line arguments, initialize a cluster resolver - Python object that tries to communicate to TPUs over network. Then we initialize our distributed TPUStrategy, and run our training code as normal.  

I also had to get rid of `tf.keras.layers.experimental.preprocessing.Hashing` layer in the model, replacing it with `tf.strings.to_hash_bucket` step in dataset preprocessing instead, because I was getting the following error:
```
ValueError: Received input tensor postId which is the output of op model/hashing_1/hash (type StringToHashBucketFast) which does not have the `_tpu_input_identity` attr. Please ensure that the inputs to this layer are taken directly from the arguments of the function called by strategy.run. Two possible causes are: dynamic batch size support or you are using a keras layer and are not passing tensors which match the dtype of the `tf.keras.Input`s.If you are triggering dynamic batch size support, you can disable it by passing tf.distribute.RunOptions(experimental_enable_dynamic_batch_size=False) to the options argument of strategy.run().
```

Now, run this code to create a tpu:
```
export TPU_ZONE=europe-west4-a
# WARNING! Billable and expensive
> yes | gcloud beta compute tpus create mgaiduk_recall1 --zone=$TPU_ZONE --version=2.11 --accelerator-type=v3-8 --project=my-first-project;
# Run this after the training to make sure you delete your TPU and stop receiving bills for it
#> yes | gcloud compute tpus delete mgaiduk_recall1 --zone=$TPU_ZONE --project=my-first-project|| true
```
This might require some additional setup - registering, quota handling and stuff like that, which I will not describe here.  

Finally, run our training code:
```
python3 train.py --tpu_name mgaiduk_recall1 --tpu_zone $TPU_ZONE -c config.yaml
```
And don't forget to shut down the tpu afterwards:
```
gcloud compute tpus delete mgaiduk_recall1 --zone=$TPU_ZONE --project=maximal-furnace-783
```

And that's it! Training code with all necessary modifications is available at [github](https://github.com/mgaiduk/mgaiduk.github.io/tree/main/my-first-book/src/code/chapter6)