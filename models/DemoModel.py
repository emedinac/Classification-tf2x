from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# A default model just for testing.
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')

        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(2)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

if __name__ == '__main__':
    # Create an instance of the model
    model = MyModel()
