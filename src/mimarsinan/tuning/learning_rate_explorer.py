class LearningRateExplorer:
    def __init__(self, training_function, evaluation_function, model, max_lr):
        self.training_function = training_function
        self.evaluation_function = evaluation_function
        self.model = model
        self.max_lr = max_lr
        self.min_lr = max_lr / 1000

    def find_lr_for_tuning(self):
        original_state = self.model.state_dict()
        original_acc = self.evaluation_function()

        lr = self.max_lr
        acc = 0
        while acc < original_acc and lr > self.min_lr:
            self.training_function(lr)
            acc = self.evaluation_function()
            if acc < original_acc:
                lr *= 0.9
            
            self.model.load_state_dict(original_state)
        
        print("  Tuning lr discovered = {}".format(lr))
        return lr
