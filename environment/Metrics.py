import tensorflow as tf

# Now the system is independent of TF2.x
# Other for analysis and inference
class Losses:
    def __init__(self):
        pass;
    def SetCrossEntropy():
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss_function

class Optimizers:
    def __init__(self):
        pass;
    def SetAdam(lr, epochs, decay=True):
        if decay:
            lr = tf.optimizers.schedules.PolynomialDecay(lr, epochs, 1e-5, 2)
        return tf.keras.optimizers.Adam(lr)

# Save and storage information using tensorboard
class Logger:
    def __init__(self, log_file='logdir', epoch=0):
        self.summary_writer = tf.summary.create_file_writer(log_file)
        self.epoch = epoch
    def SaveScalar(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name,value,self.epoch)
    def UpdateEpoch(self):
        self.epoch = self.epoch+1