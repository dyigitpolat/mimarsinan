"""Importing this package runs every provider's @register decorator."""

from . import mnist_data_provider as mnist_data_provider
from . import mnist32_data_provider as mnist32_data_provider
from . import fashion_mnist_data_provider as fashion_mnist_data_provider
from . import kmnist_data_provider as kmnist_data_provider
from . import svhn_data_provider as svhn_data_provider
from . import cifar10_data_provider as cifar10_data_provider
from . import cifar100_data_provider as cifar100_data_provider
from . import ecg_data_provider as ecg_data_provider
from . import imagenet_data_provider as imagenet_data_provider

