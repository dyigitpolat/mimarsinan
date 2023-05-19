class LearningRateExplorer:
    def __init__(self, 
                 trainer, model, max_lr, min_lr, 
                 desired_improvement = 0.0, base_accuracy = None):
        self.training_function = trainer.train_one_step
        self.evaluation_function = trainer.validate_train
        self.model = model
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.desired_improvement = desired_improvement
        self.base_accuracy = base_accuracy

    def find_lr_for_tuning(self):
        original_state = self.model.state_dict()

        if self.base_accuracy is None:
            self.base_accuracy = self.evaluation_function()

        real_improvement = self.desired_improvement * (1.0 - self.base_accuracy)
        target_acc = self.base_accuracy + real_improvement

        lr = self.max_lr
        acc = 0
        while acc < target_acc and lr > self.min_lr:
            print(f"  Trying lr = {lr}")
            self.training_function(lr)
            acc = self.evaluation_function()
            if acc < target_acc:
                lr *= 0.7
                real_improvement *= 0.999
                target_acc = self.base_accuracy + real_improvement
            
            self.model.load_state_dict(original_state)
        return lr