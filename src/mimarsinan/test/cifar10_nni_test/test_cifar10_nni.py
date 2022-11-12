from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.test.test_utils import *

from nni.experiment import Experiment

def test_cifar10_nni():
    experiment = Experiment('local')

    search_space = {

        'patch_cols': {'_type': 'quniform', '_value': [2, 8, 1]},
        'patch_rows': {'_type': 'quniform', '_value': [2, 8, 1]},
        'patch_features': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'patch_channels': {'_type': 'quniform', '_value': [1, 16, 1]},
        'mixer_features': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'inner_mlp_count': {'_type': 'quniform', '_value': [1, 5, 1]},
        'inner_mlp_width': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'patch_center_x': {'_type': 'uniform', '_value': [-0.15, 0.15]},
        'patch_center_y': {'_type': 'uniform', '_value': [-0.15, 0.15]},
        'patch_lensing_exp_x': {'_type': 'uniform', '_value': [0.5, 2.0]},
        'patch_lensing_exp_y': {'_type': 'uniform', '_value': [0.5, 2.0]}
    }

    experiment.config.trial_command = \
        'python mimarsinan/test/cifar10_nni_test/cifar10_nni_worker.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10

    experiment.run(8082, wait_completion=True)

    trials = experiment.export_data()
    for trial in trials:
        print(trial.parameter, "ntk: ", trial.value)

    print("best: ", min([(t.value, t.parameter) for t in trials]))
    