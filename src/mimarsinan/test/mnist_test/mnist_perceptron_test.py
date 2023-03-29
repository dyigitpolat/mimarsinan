from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.mnist_test.mnist_test_utils import *

def test_mnist_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_input_shape = (1, 28*28)
    mnist_output_size = 10

    ann_model = SimplePerceptronFlow(mnist_input_shape, mnist_output_size)

    print("Pretraining model...")
    pretrain_epochs = 1
    train_on_mnist(ann_model, device, pretrain_epochs)

    mapping = SoftCoreMapping()
    ann_model.get_mapper_repr().map(mapping)
    print("MNIST perceptron test done.")