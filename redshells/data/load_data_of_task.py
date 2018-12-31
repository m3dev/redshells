import gokart
import luigi


class LoadDataOfTask(gokart.TaskOnKart):
    task_namespace = 'redshells'
    data_task = gokart.TaskInstanceParameter()
    target_name = luigi.Parameter()

    def requires(self):
        return self.data_task

    def output(self):
        return self.input()[self.target_name]
