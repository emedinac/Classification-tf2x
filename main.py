import dataset
import models as m
from environment import Metrics, TrainModule, TestModule

# Config arguments or configuration file
Epochs = 20
classes=2
eval_each_num_epochs = 5
# ...

# Training environment configuration
train_stage = TrainModule()
train_stage.loss_function = Metrics.Losses.SetCrossEntropy()
train_stage.optimizer = Metrics.Optimizers.SetAdam(1e-3, Epochs)

# Testing environment configuration
test_stage = TestModule()
test_stage.loss_function = Metrics.Losses.SetCrossEntropy()

# Load dataset created in TFDS
train_ds, test_ds = dataset.GetDB(name="dataset")

# Model configuration
# model = m.MyModel() # User-defined
model = m.EfficientNetB0(classes=classes) # User-defined


storage = Metrics.Logger(log_file='logdir')
for epoch in range(Epochs):
    train_stage.reset_states()
    test_stage.reset_states()
    for image, label in train_ds:
        train_stage.train_step(model, image, label)

    # if epoch%eval_each_num_epochs == 0:
    for image, label in test_ds:
        test_stage.test_step(model, image, label)
    test_stage.Update_best_loss()

    # Save information into log tensorboard. An information batch can be added.
    storage.SaveScalar('train_loss',train_stage.train_loss.result())
    storage.SaveScalar('train_accuracy',train_stage.train_accuracy.result() * 100)
    storage.SaveScalar('test_loss',test_stage.test_loss.result())
    storage.SaveScalar('test_accuracy',test_stage.test_accuracy.result() * 100)
    storage.UpdateEpoch()
    # Save the best model weights.
    if test_stage.loss_updated:
        # model.save("test") # Init saver # Full graph
        model.save_weights("weights/{0}_{1}.pth".format(model.name, epoch))

    print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_stage.train_loss.result()}, '
    f'Accuracy: {train_stage.train_accuracy.result() * 100}, '
    f'Test Loss: {test_stage.test_loss.result()}, '
    f'Test Accuracy: {test_stage.test_accuracy.result() * 100}, '
    )
# MyModel Epoch 10, Loss: 0.4299190938472748, Accuracy: 88.12734985351562, Test Loss: 0.4228290617465973, Test Accuracy: 88.58039093017578
