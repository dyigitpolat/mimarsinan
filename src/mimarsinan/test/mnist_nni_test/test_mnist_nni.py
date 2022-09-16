from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.test.test_utils import *

from nni.experiment import Experiment

def test_mnist_nni():
    experiment = Experiment('local')

    search_space = {
        'inner_mlp_count': {'_type': 'quniform', '_value': [1, 5, 1]},
        'inner_mlp_width': {'_type': 'choice', '_value': [
            64, 128, 256, 512, 1024]}
    }

    experiment.config.trial_command = \
        'python mimarsinan/test/mnist_nni_test/mnist_nni_worker.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 40
    experiment.config.trial_concurrency = 20

    experiment.run(8081, wait_completion=True)

    for trial in experiment.export_data():
        print(trial.parameter, "accuracy: ", trial.value)
    