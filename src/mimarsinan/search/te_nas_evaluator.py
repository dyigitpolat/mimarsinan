from mimarsinan.model_evaluation.te_nas_utils import get_ntk_n
from mimarsinan.model_training.basic_trainer import BasicTrainer

class TE_NAS_Evaluator:
    def __init__(self, data_provider, loss, lr, device):
        self.data_provider = data_provider
        self.loss = loss
        self.lr = lr
        self.device = device

    def evaluate(self, model):
        trainer = BasicTrainer(
            model, 
            self.device, self.data_provider,
            self.loss)
        
        trainer.train_one_step(self.lr)
        
        ntk_score = get_ntk_n(
            self.data_provider.get_validation_loader(
                self.data_provider.get_validation_batch_size()
            ), 
            [model],
            self.device)[0]
        
        return 1.0 / ntk_score
