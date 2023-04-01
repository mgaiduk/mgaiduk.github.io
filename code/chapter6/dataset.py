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
                if feature.hash:
                    t = tf.strings.to_hash_bucket(t, feature.vocab_size)
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
