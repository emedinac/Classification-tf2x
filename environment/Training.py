from .Metrics import *

class TrainModule:
    def __init__(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
    @tf.function
    def reset_states(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
    @tf.function
    def train_step(self, model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

if __name__ == '__main__':
    pass; # UnitTest modules
