from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.test.test_utils import *

from nni.experiment import Experiment

def test_mnist_ntk_nni():
    experiment = Experiment('local')

    search_space = {
        'inner_mlp_count': {'_type': 'quniform', '_value': [1, 10, 1]},
        'inner_mlp_width': {'_type': 'choice', '_value': [
            32, 64, 128, 256, 512, 1024]}
    }

    experiment.config.trial_command = \
        'python mimarsinan/test/mnist_ntk_nni_test/mnist_ntk_nni_worker.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10

    experiment.run(8080, wait_completion=True)

    for trial in experiment.export_data():
        print(trial.parameter, "accuracy: ", trial.value)
    