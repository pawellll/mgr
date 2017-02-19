from os import listdir
from os.path import isfile, join
import tensorflow as tf
import skimage.io
import numpy
from Configuration import Configuration as Config


class DataProvider:
    def __init__(self, train_folder, submission_folder):
        self._train_folder = train_folder
        self._submission_folder = submission_folder

        self._submission_file_paths = list()

        self._train_image_files = list()
        self._train_labels = list()

        self._test_image_files = list()
        self._test_labels = list()

        self.load_train_data_info()
        self.load_submission_data_info()

    def extract_image_file_names_with_labels(self):
        files = self.files_in_path(self._train_folder)

        image_files = []
        image_labels = []

        for imageFilePath in files:
            label_name = imageFilePath[0:3]

            label = [0, 0]

            if label_name == "cat":
                label[0] = 1
            elif label_name == "dog":
                label[1] = 1

            image_files.append(self._train_folder + "/" + imageFilePath)
            image_labels.append(label)

        return image_files, image_labels

    def test_data_batch(self):
        test_image_files = self._test_image_files[0:Config.batch_size]
        test_labels = self._test_labels[0:Config.batch_size]

        self._test_image_files = self._test_image_files[Config.batch_size:]
        self._test_labels = self._test_labels[Config.batch_size:]

        images = list()

        for file_path in test_image_files:
            image = self.load_image(file_path)
            images.append(image)

        return numpy.array(images), numpy.array(test_labels)


    def next_data_batch(self):
        if len(self._train_image_files) < Config.batch_size:
            self.load_train_data_info()

        train_image_files = self._train_image_files[0:Config.batch_size]
        train_labels = self._train_labels[0:Config.batch_size]

        self._train_image_files = self._train_image_files[Config.batch_size:]
        self._train_labels = self._train_labels[Config.batch_size:]

        images = list()

        for file_path in train_image_files:
            image = self.load_image(file_path)
            images.append(image)

        return numpy.array(images), numpy.array(train_labels)

    def load_train_data_info(self):
        image_files, labels = self.extract_image_file_names_with_labels()

        data_set_size = len(image_files)
        train_number = data_set_size - Config.test_number

        self._train_image_files = image_files[0:train_number]
        self._train_labels = labels[0:train_number]

        self._test_image_files = image_files[train_number + 1: data_set_size - 1]
        self._test_labels = labels[train_number + 1: data_set_size - 1]

    def load_submission_data_info(self):
        for i in range(1, 125001):
            file_path = self._submission_folder + "/" + str(i) + ".jpg"
            self._submission_file_paths.append(file_path)

    def submission_data_batch(self, batch_size):
        sliced_list = self._submission_file_paths[:batch_size]
        self._submission_file_paths = self._submission_file_paths[batch_size:]

        images = list()

        for file_path in sliced_list:
            image = self.load_image(file_path)
            images.append(image)

        return numpy.array(images), sliced_list

    @staticmethod
    def files_in_path(path):
        return [f for f in listdir(path) if isfile(join(path, f))]

    @staticmethod
    def flatten(matrix):
        return [item for vector in matrix for item in vector]

    @staticmethod
    def add_noise(image):
        # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
     # 	distorted_image = tf.image.random_brightness(image, max_delta=0.3)
           
    	# distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.5)

     # 	with tf.Session() as sess:
    	#     image = distorted_image.eval()

        return image  

    @staticmethod
    def load_image(image_file):
        image = skimage.io.imread(image_file).astype(numpy.float32)
        image = DataProvider.add_noise(image)
        image = DataProvider.flatten(image)
        image[:] = [value / 255.0 for value in image]
        return image


