import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D
from tensorflow.keras import Model

import dataset

test_summary_writer = tf.summary.create_file_writer('logdir')

# Data augmentation here may increase the accuracy in the training results.
def Augmentation(image):
    image = tf.image.resize(image,(224,224))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    # image = tf.image.resize_with_crop_or_pad(image,20,20)
    # image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    return image

# Load my dataset here
# This functions were created in order to follow the full-pipeline format from TFDS-nightly.
train_ds = tfds.load("dataset", split='train', shuffle_files=True, batch_size=8)
test_ds = tfds.load("dataset", split='test', shuffle_files=True, batch_size=8)
train_ds = train_ds.map(lambda data: (Augmentation(data["image"]), data["label"])).cache()
test_ds = test_ds.map(lambda data: (Augmentation(data["image"]), data["label"])).cache()

# A default model just for testing.
# class MyModel(Model):
#   def __init__(self):
#     super(MyModel, self).__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')

#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)
# Create an instance of the model
# model = MyModel()

base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", classes=2)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_stage(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_stage(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(20):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for image, label in train_ds:
        train_stage(image, label)

    for image, label in test_ds:
        test_stage(image, label)

    with test_summary_writer.as_default():
        tf.summary.scalar('train_loss',train_loss.result(),epoch)
        tf.summary.scalar('train_accuracy',train_accuracy.result() * 100,epoch)
        tf.summary.scalar('train_accuracy',test_loss.result(),epoch)
        tf.summary.scalar('train_accuracy',test_accuracy.result() * 100,epoch)
    print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
    )