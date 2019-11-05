import numpy as np


class ReadFile:
    """
    Class implementation for reading ascii-based file format images.
    """

    def __init__(self, file):
        """
        Read the input .txt file and removes the first two lines of rubbish comments.
        :param file: The file to be read.
        """
        self.file = open(file, "r")
        self.file.readline()
        self.file.readline()

    def read_images(self):
        """
        Reads the third line from the image.txt file and extracts the values needed to understand
        the properties of the images that are stored later in the file. Then stores all images into
        a numpy array.
        :return: An array of images.
        """
        images, rows, cols, digits = map(int, self.file.readline().split())

        img = []
        for i in range(images):
            line = list(map(int, self.file.readline().split()))
            line = np.array(list(map(0.001.__mul__, line)))
            img.append(line)
        img = np.array(img)

        return img, rows, cols, digits

    def read_label(self):
        """
        Reads the third line from the label.txt file and extracts the values needed to understand
        the properties of the labels that are stored later in the file. Then stores all labels into
        a numpy array.
        :return: An array of labels.
        """
        images, digits = map(int, self.file.readline().split())

        # Read the file and store every row (image) as an array inside an array
        label = []
        for _ in range(images):
            line = self.file.readline().strip()
            label.append(int(line))
        label = np.array(label)

        return label

    def close_file(self):
        """
        Close down the file after we have all the data we need.
        """
        self.file.close()
