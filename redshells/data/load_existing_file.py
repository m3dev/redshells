import gokart
import luigi


class LoadExistingFile(gokart.TaskOnKart):
    task_namespace = 'redshells'
    file_path = luigi.Parameter()  # type: str

    def output(self):
        self.workspace_directory = ''
        return self.make_target(self.file_path, use_unique_id=False)

    def run(self):
        raise RuntimeError(f'{self.file_path} does not exist.')
