import wandb
import time

class WandB_Reporter:
    initialized = False
    def __init__(self, project_name, experiment_name):
        if not WandB_Reporter.initialized:
            wandb.login()
            WandB_Reporter.initialized = True
        else:
            wandb.finish()
            
        wandb.init(project=project_name, name=experiment_name)

        self.report_timestamps = {}
        self.reporting_intervals = {}

    def console_log(self, metric_name, metric_value):
        if metric_name not in self.report_timestamps:
            self.report_timestamps[metric_name] = 0
        
        if metric_name not in self.reporting_intervals:
            self.reporting_intervals[metric_name] = 0.5
        
        current_timestamp = time.time()
        current_interval = current_timestamp - self.report_timestamps[metric_name]

        if current_interval > self.reporting_intervals[metric_name]:
            print(f"            {metric_name}: {metric_value}")
            self.report_timestamps[metric_name] = current_timestamp

            if current_interval < 5.0:
                self.reporting_intervals[metric_name] *= 1.5
            
            if current_interval > 5.0:
                self.reporting_intervals[metric_name] *= 0.75

    def report(self, metric_name, metric_value, step = None):
        wandb.log({metric_name: metric_value}, step = step)
        self.console_log(metric_name, metric_value)