import luigi

import gokart


class CalculateDocumentEmbedding(gokart.TaskOnKart):
    """
    Calculate document embeddings 
    """
    task_namespace = 'redshells.word_item_similarity'
    document_task = gokart.TaskInstanceParameter()
    scdv_task = gokart.TaskInstanceParameter()
    item_id_column_name = luigi.Parameter()  # type: str
    document_column_name = luigi.Parameter()  # type: str
    l2_normalize = luigi.BoolParameter()  # type: bool
    output_file_path = luigi.Parameter(default='app/word_item_similarity/calculate_document_embedding.pkl')  # type: str

    def requires(self):
        return dict(document=self.document_task, scdv=self.scdv_task)

    def output(self):
        return self.make_target(self.output_file_path)

    def run(self):
        scdv = self.load('scdv')
        document = self.load_data_frame(
            'document', required_columns={self.item_id_column_name, self.document_column_name})

        documents = document[self.document_column_name].tolist()
        embeddings = scdv.infer_vector(documents, l2_normalize=self.l2_normalize)
        self.dump(dict(zip(document[self.item_id_column_name].tolist(), list(embeddings))))
