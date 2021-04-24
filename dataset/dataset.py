"""dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import csv # Temporal csv reader
# TODO(dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(dataset): BibTeX citation
_CITATION = """
"""


class Dataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(40, 40, 3)),
            'label': tfds.features.ClassLabel(names=['malaria', 'no malaria']), # num_classes=2),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # This section
    path = "dataset ML candidates/" # Set this manually inside dataset folder to avoid automation
    if not tf.io.gfile.exists(path + 'training.csv'): self.ext = "../"
    else: self.ext = ""
    return {
        'train': self._generate_examples(data_path=self.ext + path + 'training.csv'),
        'test': self._generate_examples(data_path=self.ext + path + 'validation.csv'),
    }

  def _generate_examples(self, data_path):
    # Read the input data out of the source files
    with tf.io.gfile.GFile(data_path) as f:
      for row in csv.DictReader(f):
        image_id = int(row['']) # using idx as image_id - Also temporal
        yield image_id, {
            'image': self.ext +row['imgs'],
            'label': row['labels'],
        }