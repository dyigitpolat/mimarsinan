from init import *

from mimarsinan.test.mnist_test.mnist_patched_perceptron_test import *
from mimarsinan.test.cifar10_test.cifar10_patch_perceptron_test import *
from mimarsinan.test.ecg_test.ecg_test import *


def main():
    init()

    dataset_selection = input("Select dataset (mnist, cifar10, ecg): ")

    if dataset_selection[0] == 'mnist'[0]:
        test_mnist_patched_perceptron()
    elif dataset_selection[0] == 'cifar10'[0]:
        test_cifar10_patched_perceptron()
    elif dataset_selection[0] == 'ecg'[0]:
        test_ecg_patched_perceptron()

if __name__ == "__main__":
    main()