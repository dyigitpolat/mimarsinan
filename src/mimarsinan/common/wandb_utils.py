import wandb

class WandB_Reporter:
    def __init__(self, project_name, experiment_name):
        wandb.login()
        wandb.init(project=project_name, name=experiment_name)
    
    def report(self, metric_name, metric_value, step = None):
        wandb.log({metric_name: metric_value}, step = step)