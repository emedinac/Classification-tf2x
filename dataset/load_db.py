import data_processing as proc
import tensorflow_datasets as tfds

def GetDB(name):
    # Load my dataset here
    # This functions were created in order to follow the full-pipeline format from TFDS-nightly.
    train_ds = tfds.load(name, split='train', shuffle_files=True, batch_size=8)
    test_ds = tfds.load(name, split='test', shuffle_files=True, batch_size=8)
    train_ds = train_ds.map(lambda data: (proc.Augmentation(data["image"]), data["label"])).cache()
    test_ds = test_ds.map(lambda data: (proc.Augmentation(data["image"]), data["label"])).cache()
    return train_ds, test_ds
