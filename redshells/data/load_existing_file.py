import gokart
import luigi


class LoadExistingFile(gokart.TaskOnKart):
    file_path = luigi.Parameter()  # type: str
    workspace_directory = ''

    def output(self):
        return self.make_target(self.file_path, use_unique_id=False)

    def run(self):
        raise RuntimeError(f'{self.file_path} does not exist.')
