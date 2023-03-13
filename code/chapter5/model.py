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
