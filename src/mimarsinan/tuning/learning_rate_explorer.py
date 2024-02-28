import copy
class LearningRateExplorer:
    def __init__(self, 
        clone_state,
        restore_state,
        trainer, model, max_lr, min_lr, 
        desired_improvement = 0.0, base_accuracy = None):

        self.clone_state = clone_state
        self.restore_state = restore_state

        self.training_function = trainer.train_one_step
        self.evaluation_function = trainer.validate

        self.model = model
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.desired_improvement = desired_improvement
        self.base_accuracy = base_accuracy

    def find_lr_for_tuning(self):
        original_state = self.clone_state()
        print("original acc:", self.evaluation_function())

        if self.base_accuracy is None:
            self.base_accuracy = self.evaluation_function()

        iters = 100
        lr = self.max_lr
        lrs = []
        accs = []
        while lr > self.min_lr:
            acc = 0
            iter = 0
            while acc < self.base_accuracy and iter < iters:
                self.training_function(lr)
                acc = self.evaluation_function()
                iter += 1

            lrs.append(lr)
            accs.append(acc)
            lr *= 0.7
            self.restore_state(original_state)

        print (lrs)
        print (accs)
        best_lr = lrs[accs.index(max(accs))]       
        return best_lr