import gokart
import luigi

import redshells.data
import redshells.train


class TrainSCDVWithExistingFile(gokart.TaskOnKart):
    task_namespace = 'examples'
    text_data_file_path = luigi.Parameter()  # type: str

    def requires(self):
        text_data = redshells.data.LoadExistingFile(file_path=self.text_data_file_path)
        dictionary = redshells.train.TrainDictionary(tokenized_text_data_task=text_data)
        fasttext = redshells.train.TrainFastText(tokenized_text_data_task=text_data)
        scdv = redshells.train.TrainSCDV(
            tokenized_text_data_task=text_data, dictionary_task=dictionary, word2vec_task=fasttext)
        return scdv

    def output(self):
        return self.input()


if __name__ == '__main__':
    # Please put sample_tokenized_data.pkl on redshells/examples/resources/.
    # "sample_tokenized_data.pkl" must have type "List[List[str]]"
    luigi.run([
        'examples.TrainSCDVWithExistingFile', '--text-data-file-path', './resources/sample_tokenized_data.pkl',
        '--local-scheduler'
    ])
