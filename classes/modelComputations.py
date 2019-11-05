import numpy as np
from classes.network import Network


def create_networks(labels, net_size):
    """
    Create different networks based on the labels (digits) in the image file.
    :param labels: The labels for which we will create networks, need to cast to str to iterate and then to int for
    the Network class.
    :param net_size: The size of the weight list the network should have.
    :return: A list of testing images and a list of training images
    """
    labels = str(labels)
    networks = []
    for num in labels:
        networks.append(Network(int(num), net_size))

    return networks


def list_splitter(list_to_split, ratio):
    """
    Split the list of all images into test and training sub sets.
    :param list_to_split: Numpy array of all the images of handwritten numbers
    :param ratio: The proportion between the test and training sub set. Sets the midpoint for the split.
    :return: A list of testing images and a list of training images
    """
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]


def list_shuffler(image_list_to_shuffle, label_to_shuffle):
    """
    Takes in two lists of equal size and shuffles each list randomly. However both lists
    are shuffled in the same random order.
    :param image_list_to_shuffle: List of images
    :param label_to_shuffle: List of labels for the images.
    :return:
    """
    temp_zip = list(zip(image_list_to_shuffle, label_to_shuffle))
    np.random.shuffle(temp_zip)
    shuffled_image_list, shuffled_label_list = zip(*temp_zip)
    return shuffled_image_list, shuffled_label_list


def compute_highest(number_list, networks):
    """
    A function that finds the highest value in a list of values,
    then takes the index where that value lies and retrieve the label
    that are on the same index in the list of networks.
    :param number_list: List with values that each network have calculated
    :param networks: List containing all network
    :return: The correct label of the network with the highest calculation
    """
    return networks[number_list.index(max(number_list))].label


def train_model(nets, training_images, training_labels, test_images, test_labels):
    """
    A function that updates the weights of each neuron and then classifies unseen images to
    evaluate if the models if trained enough.
    :param nets: The neurons in the network.
    :param training_images: Images to train the model on.
    :param training_labels: The labels for each training image.
    :param test_images: Images to test the model on.
    :param test_labels: Labels of the unseen images, to test if the model predicts correctly.
    :return: The % of correct classifications the model on test images after training.
    """
    mean_error = 1
    correctly_classified = 0
    while mean_error > 0.2:  # Train the model until an error of less than 0.2 is found.

        # Update the weights for all the neurons, aka training the model
        for i, image in enumerate(training_images):
            for net in nets:
                act = net.activation_function(net.dot_product(image))
                err = net.calculate_error(training_labels[i], act)
                net.calculate_new_weight(err, image, 0.045)

        nets_ans = [0 for _ in nets]
        total_correct_ans = 0

        # Compare the current model predictions the real, unseen images.
        # Calculate error to see if the model has reached a satisfactory error
        error = 0
        for k, img in enumerate(test_images):
            for j, net in enumerate(nets):
                nets_ans[j] = net.activation_function(net.dot_product(img))
                error += np.abs(net.calculate_error(test_labels[k], nets_ans[j]))

            # if the predicted label is the same as the correct label
            if compute_highest(nets_ans, nets) == test_labels[k]:
                total_correct_ans += 1

        correctly_classified = (total_correct_ans / len(test_labels)) * 100
        mean_error = error / (len(test_images) * len(nets))

    return correctly_classified


def classify_validation_images(nets, val_all_images):
    """
    Classify unseen images and store the model predictions in a numpy array.
    :param nets: The neurons in the network.
    :param val_all_images: The unseen validation images.
    """
    val_ans = [0 for _ in nets]
    label = []

    for k, img in enumerate(val_all_images):
        for j, net in enumerate(nets):
            dot_p = net.dot_product(img)
            val_ans[j] = net.activation_function(dot_p)

        label.append(compute_highest(val_ans, nets))
    label = np.array(label)

    return label


def compare_classified_and_validation(cl, vl):
    """
    Take a list of predicted labels the model has classified and compares those labels
    to the true labels provided.
    :param cl: Classified labels.
    :param vl: Validation labels.
    :return: The percent of labels that the model correctly classified.
    """
    h = 0
    for i, label in enumerate(cl):
        if label == vl[i]:
            h += 1
    return (100.0 * h) / len(cl)
