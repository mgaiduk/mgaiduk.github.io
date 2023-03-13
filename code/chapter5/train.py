import argparse
from config import Config
from dataset import create_dataset
from model import Model
import tensorflow as tf
import sys
print("tf.__version__: ", tf.__version__)

def save_string_gcs(string_object, gcs_dir, filename):
    string_string = json.dumps(string_object)
    with open(filename, "w") as f:
        f.write(string_string)
    os.system(f"gsutil -m cp {filename} {gcs_dir}/{filename}")
    os.system(f"rm {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config")
    args = parser.parse_args()

    config = Config(args.config)
    strategy = tf.distribute.get_strategy()
    train_dataset = create_dataset(config, strategy, config.train_path)

    with strategy.scope():
        model = Model(config)
        model.compile(model.optimizer, steps_per_execution=1)
        train_dataset = create_dataset(config, strategy, config.train_path)
        trainval_dataset = create_dataset(config, strategy, config.validate_path)
        eval_dataset = create_dataset(config, strategy, config.validate_path)
        train_steps_per_epoch = config.train_rows // config.global_batch_size
        trainval_steps_per_epoch = config.trainval_rows // config.global_batch_size
        eval_steps_per_epoch = config.eval_rows // config.global_batch_size
        checkpoints_cb = tf.keras.callbacks.ModelCheckpoint(config.save_model_path  + '/checkpoints/',  save_freq = train_steps_per_epoch//3)
        callbacks=[checkpoints_cb]
        history = model.fit(train_dataset, epochs=config.epochs, callbacks=[callbacks], steps_per_epoch=train_steps_per_epoch,
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


if __name__ == "__main__":
    main()