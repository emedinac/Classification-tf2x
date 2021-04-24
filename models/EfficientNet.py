import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model

class EfficientNetB0(Model):
    def __init__(self,classes=2):
        super(EfficientNetB0, self).__init__()
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet")
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
    def call(self, x):
        return self.model(x)
if __name__ == '__main__':
    # Create an instance of the model
    model = EfficientNetB0()
