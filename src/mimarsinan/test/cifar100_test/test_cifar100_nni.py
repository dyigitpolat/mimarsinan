from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.models.ensemble_mlp_mixer import *
from mimarsinan.test.cifar100_test.cifar100_test_utils import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar100_test.cifar100_nni_worker import get_number_of_mlp_mixers

from nni.experiment import Experiment

def test_cifar100_nni():
    experiment = Experiment('local')

    search_space = get_omihub_mlp_mixer_search_space()

    number_of_mlp_mixers = get_number_of_mlp_mixers()
    augmented_search_space = {}
    for i in range(number_of_mlp_mixers):
        for k in search_space:
            augmented_search_space[k + str(i)] = search_space[k]

    experiment.config.trial_command = \
        'python mimarsinan/test/cifar100_test/cifar100_nni_worker.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = augmented_search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
    experiment.config.max_trial_number = 500
    experiment.config.trial_concurrency = 2

    experiment.run(8080, wait_completion=True)

    trials = experiment.export_data()
    for trial in trials:
        print(trial.parameter, "ntk: ", trial.value)

    import json
    best = min([(t.value, json.dumps(t.parameter)) for t in trials])
    print("best: ", best)

    with wandb.init(project='mlp_mixer_seed_run', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        ann_model = EnsembleMLPMixer(
            get_parameter_dict_list(
                json.loads(best[1]), number_of_mlp_mixers), 32, 32, 3, 100)
        trainer = Trainer(ann_model, args)
        trainer.fit(train_dl, test_dl)
    