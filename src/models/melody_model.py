import tensorflow as tf
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.tools.data.batcher import MelodyBatcher


def build_model(features, labels, mode):
    # Input Layer
    input_layer = tf.transpose(features['x'], perm=[0, 2, 1])
    input_layer = tf.reshape(input_layer, [-1, 85, 304, 1])

    # f0
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=30,
        kernel_size=(46, 96),
        strides=(1, 1),
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 209), strides=(2, 1))

    # f1
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=60,
        kernel_size=(5, 1),
        strides=(1, 1),
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 1), strides=(2, 1))

    # f2
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=800,
        kernel_size=(8, 1),
        strides=(1, 1),
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=conv3, rate=0.5, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # f3
    conv4 = tf.layers.conv2d(
        inputs=dropout,
        filters=2,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None)

    logits = tf.reshape(conv4, [-1, 2])
    logits = tf.nn.softmax(logits, name="softmax_tensor")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": logits
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.log_loss(labels=labels, predictions=logits)
    loss = tf.identity(loss, name="loss")

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


@click.command()
@click.argument('name', type=click.STRING)  # Dataset name
def main(name):
    # Create batcher
    dataset_path = f'{project_dir}/data/processed/{name}'.replace('\\', '/')
    train_batcher = MelodyBatcher(f'{dataset_path}/train', buffer_size=1000)
    validation_batcher = MelodyBatcher(f'{dataset_path}/validation', buffer_size=500)

    # Build model
    melody_classifier = tf.estimator.Estimator(model_fn=build_model, model_dir="../../models/melody")
    tensors_to_log = {
        "probabilities": "softmax_tensor",
        "loss": "loss"
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
    evaluator_hook = tf.contrib.estimator.InMemoryEvaluatorHook(melody_classifier, input_fn=validation_batcher.get_batches, every_n_iter=50)

    melody_classifier.train(input_fn=train_batcher.get_batches, steps=10000, hooks=[logging_hook, evaluator_hook])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
