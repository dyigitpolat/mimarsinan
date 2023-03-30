from mimarsinan.test.xor_test.xor_test import *
from mimarsinan.test.mnist_test.mnist_test import *
from mimarsinan.test.mnist_nni_test.test_mnist_nni import *
from mimarsinan.test.mnist_ntk_nni_test.test_mnist_ntk_nni import *
from mimarsinan.test.cifar10_test.cifar10_test import *
from mimarsinan.test.cifar10_nni_test.test_cifar10_nni import *
from mimarsinan.test.cifar100_test.cifar100_test import *
from mimarsinan.test.cifar100_test.test_cifar100_nni import *
from mimarsinan.test.mapping_test.test_mapping import *
from mimarsinan.test.core_flow_test.core_flow_test import *
from mimarsinan.test.mnist_test.mnist_perceptron_test import *
from mimarsinan.test.mnist_test.mnist_patched_perceptron_test import *
from mimarsinan.test.cifar10_test.cifar10_patch_perceptron_test import *

from mimarsinan.test.debug_spikes_test.debug_spikes_test import *

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()

# test_debug_spikes()
# test_xor()
# test_mnist()
# test_mnist_nni()
# test_mnist_ntk_nni()
# test_cifar10()
# test_cifar10_nni()
# test_cifar100()
# test_cifar100_nni()
# test_mapping()
# test_core_flow()
# test_mnist_perceptron()
# test_mnist_patched_perceptron()
test_cifar10_patched_perceptron()