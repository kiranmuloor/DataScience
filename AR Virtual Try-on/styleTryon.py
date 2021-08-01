import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class styleTryon:
    def __init__(self, filename):
        self.filename = filename

    def objectTryon(self):
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = 1

        if result == 1:
            prediction = 'dog'
            return [{"image": prediction}]
        else:
            prediction = 'cat'
            return [{"image": prediction}]
