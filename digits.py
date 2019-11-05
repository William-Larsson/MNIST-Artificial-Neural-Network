"""
    Author: William Larsson
    Date: 2019-11-05
"""
import sys
from classes.modelComputations import *
from classes.readFile import ReadFile


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: digits.py <training images file> <training labels file>" +
              "<validation images file> <validation labels file>")
        exit()

    all_images = ReadFile(sys.argv[1])
    all_label = ReadFile(sys.argv[2])
    validation_images = ReadFile(sys.argv[3])
    validation_labels = ReadFile(sys.argv[4])

    np_all_images, img_rows, img_cols, network_labels = all_images.read_images()
    np_all_labels = all_label.read_label()
    val_all_images, val_rows, val_cols, val_network_labels = validation_images.read_images()
    val_all_labels = validation_labels.read_label()

    # Close down all the text files.
    all_images.close_file()
    all_label.close_file()
    validation_images.close_file()
    validation_labels.close_file()

    # Split and shuffle the images and labels to randomize the order
    training_images, test_images = list_splitter(np_all_images, 0.835)
    training_labels, test_labels = list_splitter(np_all_labels, 0.835)
    training_images, training_labels = list_shuffler(training_images, training_labels)
    test_images, test_labels = list_shuffler(test_images, test_labels)

    # Array of neurons, one for each number in the given MNIST data subset.
    nets = create_networks(network_labels, img_rows*img_cols)

    # train the network and classify unseen images
    training_accuracy = train_model(nets, training_images, training_labels, test_images, test_labels)
    classified_labels = classify_validation_images(nets, val_all_images)
    classification_accuracy = compare_classified_and_validation(classified_labels, val_all_labels)

    print("Correct classification of the test images:        " + str(round(training_accuracy,2)) + "%")
    print("Correct classification of the validation images:  " + str(round(classification_accuracy,2)) + "%")
