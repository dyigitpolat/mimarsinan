import wandb

class WandB_Reporter:
    initialized = False
    def __init__(self, project_name, experiment_name):
        if not WandB_Reporter.initialized:
            wandb.login()
            WandB_Reporter.initialized = True
        else:
            wandb.finish()
            
        wandb.init(project=project_name, name=experiment_name)

    def report(self, metric_name, metric_value, step = None):
        wandb.log({metric_name: metric_value}, step = step)