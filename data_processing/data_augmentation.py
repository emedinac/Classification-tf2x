import tensorflow as tf

# Data augmentation here may increase the accuracy in the training results.
def Augmentation(image):
    image = tf.image.resize(image,(224,224))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    # image = tf.image.resize_with_crop_or_pad(image,20,20)
    # image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    return image
