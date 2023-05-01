class LearningRateExplorer:
    def __init__(self, trainer, model, max_lr, min_lr, desired_improvement = 0.0):
        self.training_function = trainer._train_one_step
        self.evaluation_function = trainer.validate_train
        self.model = model
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.desired_improvement = desired_improvement

    def find_lr_for_tuning(self):
        original_state = self.model.state_dict()
        original_acc = self.evaluation_function()

        lr = self.max_lr
        acc = 0
        while acc < original_acc and lr > (self.min_lr * (1 + self.desired_improvement)):
            self.training_function(lr)
            acc = self.evaluation_function()
            if acc < original_acc:
                lr *= 0.9
            
            self.model.load_state_dict(original_state)
        return lr
