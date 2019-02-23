import os
from logging import getLogger
from typing import Any, Dict

import luigi

import gokart
import redshells.app.word_item_similarity
import redshells.data
import redshells.train

logger = getLogger(__name__)


class WordItemSimilarityConfig(luigi.Config):
    task_namespace = 'redshells.word_item_similarity'
    matrix_factorization_kwargs = luigi.DictParameter(
        default=dict(
            n_latent_factors=20,
            learning_rate=1e-3,
            reg_item=1e-5,
            reg_user=1e-5,
            batch_size=2**10,
            epoch_size=30,
            test_size=0.1,
        ))  # type: Dict[str, Any]

    xgb_classifier_kwargs = luigi.DictParameter(
        default=dict(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=300,
            silent=True,
            objective="binary:logistic",
            booster='gbtree',
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5))  # type: Dict[str, Any]

    dictionary_filter_kwargs = luigi.DictParameter(
        default=dict(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None))  # type: Dict[str, Any]

    fasttext_kwargs = luigi.DictParameter(
        default=dict(
            corpus_file=None,
            sg=0,
            hs=0,
            size=200,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            word_ngrams=1,
            sample=1e-3,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            iter=5,
            null_word=0,
            min_n=3,
            max_n=6,
            sorted_vocab=1,
            bucket=2000000,
            trim_rule=None))  # type: Dict[str, Any]

    scdv_kwargs = luigi.DictParameter(default=dict(
        cluster_size=60,
        sparsity_percentage=0.04,
    ))  # type: Dict[str, Any]

    similarity_model_kwargs = luigi.DictParameter(
        default=dict(
            max_input_data_size=500000,
            valid_embedding_size=1000,
            prequery_return_size=10000,
            return_size=100,
        ))  # type: Dict[str, Any]


class BuildWordItemSimilarity(gokart.TaskOnKart):
    """ Calculate similarities between keywords and items.
    This task builds a model as following way:

    * Calculate similarities between items using a matrix factorization method.
    * Calculate similarities between items using keyword matching.
    * Calculate document embeddings using the SCDV.
    * Train XGBoost to predict similarities using elementwise product of document embeddings as input features.
    *
    """
    task_namespace = 'redshells.word_item_similarity'
    word_data_task = gokart.TaskInstanceParameter(description='A task which outputs `List[str]`.')
    item_train_data_task = gokart.TaskInstanceParameter(
        description='A task which outputs `pd.DataFrame` with columns=["item_id", "token", "title_token"].')
    click_data_task = gokart.TaskInstanceParameter(
        description='A task which outputs `pd.DataFrame` with columns=["user_id", "item_id", "service_id"].')
    item_predict_data_task = gokart.TaskInstanceParameter(
        description='A task which outputs `pd.DataFrame` with columns=["item_id", "token", "title_token"].')
    text_data_task = gokart.TaskInstanceParameter(
        description='A task which outputs `List[List[str]]` for FastText training.')
    use_only_title = luigi.BoolParameter(default=False)  # type: bool
    word_embedding_type = luigi.Parameter(
        default='average', description='A type of word embedding in prediction. This must be "average" or "word"')  # type: str

    def __init__(self, *args, **kwargs) -> None:
        super(BuildWordItemSimilarity, self).__init__(*args, **kwargs)
        self.scdv = None
        self.word2items = None
        self.word2embedding = None
        self.item2embedding = None
        self.similarity_train_data = None
        self.similarity_model = None
        self.word2average_embedding = None
        self.predict_item2embedding = None
        self.reduced_word2embedding = None
        self.item2title_embedding = None
        self.filtered_word2items = None

    def requires(self):
        word_data = self.word_data_task
        item_train_data = self.item_train_data_task
        click_data = self.click_data_task
        item_predict_data = self.item_predict_data_task
        text_data = self.text_data_task

        tfidf = redshells.train.TrainTfidf(
            tokenized_text_data_task=redshells.data.data_frame_utils.ExtractColumnAsList(data_task=item_train_data, column_name='token'))

        keyword_item_data = redshells.app.word_item_similarity.FindItemKeywordByMatching(
            target_keyword_task=word_data,
            item_task=item_train_data,
            item_id_column_name='item_id',
            item_keyword_column_name='token',
            tfidf_task=tfidf,
            keep_top_rate=0.3)
        self.word2items = redshells.data.data_frame_utils.GroupByColumnAsDict(
            data_task=keyword_item_data,
            key_column_name='keyword',
            value_column_name='item_id')

        # train similarity model
        self.scdv = self._train_scdv(item_train_data, text_data)
        self.word2embedding = self._calculate_word2embedding(self.scdv, word_data)
        self.item2embedding = self._calculate_item2embedding(self.scdv, self.word2embedding, item_train_data, use_only_title=self.use_only_title)
        self.item2title_embedding = self._calculate_item2embedding(self.scdv, self.word2embedding, item_train_data, use_only_title=True)
        self.similarity_train_data = self._calculate_similarity_train_data(click_data, self.word2items)
        self.similarity_model = self._train_similarity_model(self.item2embedding, self.similarity_train_data)
        self.reduced_word2embedding = self._reduce_dimension(self.word2embedding, self.word2embedding)
        self.filtered_word2items = self._filter_items(self.word2items, self.reduced_word2embedding, self.item2title_embedding, no_below=0.1)

        # calculate similarity
        self.word2average_embedding = self._calculate_word2average_embedding(word_data, self.filtered_word2items, self.item2embedding)
        self.predict_item2embedding = self._calculate_item2embedding(self.scdv, self.word2embedding, item_predict_data, use_only_title=self.use_only_title)
        embedding_map = {'word': self.reduced_word2embedding, 'average': self.word2average_embedding, 'item': self.item2embedding}
        similarity = self._calculate_similarity(self.similarity_model, embedding_map[self.word_embedding_type], self.predict_item2embedding)
        return similarity

    def output(self):
        return self.input()

    def _calculate_word2average_embedding(self, word_data, word2items, item2embedding):
        word2average_embedding = redshells.app.word_item_similarity.CalculateWordEmbedding(
            word_task=word_data, word2item_task=word2items, item2embedding_task=item2embedding)
        return word2average_embedding

    def _calculate_similarity(self, similarity_model, word2average_embedding, predict_item2embedding):
        config = WordItemSimilarityConfig()
        similarity = redshells.app.word_item_similarity.CalculateWordItemSimilarity(
            word2embedding_task=word2average_embedding,
            item2embedding_task=predict_item2embedding,
            similarity_model_task=similarity_model,
            prequery_return_size=config.similarity_model_kwargs['prequery_return_size'],
            return_size=config.similarity_model_kwargs['return_size'])
        return similarity

    def _train_similarity_model(self, item2embedding, similarity_data):
        config = WordItemSimilarityConfig()
        similarity_data = redshells.data.data_frame_utils.SampleData(
            data_task=similarity_data, sample_size=config.similarity_model_kwargs['max_input_data_size'])
        similarity_model = redshells.train.TrainPairwiseSimilarityModel(
            item2embedding_task=item2embedding,
            similarity_data_task=similarity_data,
            item0_column_name='item_id_0',
            item1_column_name='item_id_1',
            similarity_column_name='similarity',
            model_name='XGBClassifier',
            model_kwargs=config.xgb_classifier_kwargs)
        return similarity_model

    def _train_scdv(self, item_train_data, text_data):
        config = WordItemSimilarityConfig()
        column_name = 'title_token' if self.use_only_title else 'token'
        item_text_data = redshells.data.data_frame_utils.ExtractColumnAsList(
            data_task=item_train_data, column_name=column_name)
        dictionary = redshells.train.TrainDictionary(
            tokenized_text_data_task=item_text_data, dictionary_filter_kwargs=config.dictionary_filter_kwargs)
        fasttext = redshells.train.TrainFastText(
            tokenized_text_data_task=text_data, fasttext_kwargs=config.fasttext_kwargs)
        scdv = redshells.train.TrainSCDV(
            tokenized_text_data_task=item_text_data,
            dictionary_task=dictionary,
            word2vec_task=fasttext,
            cluster_size=config.scdv_kwargs['cluster_size'],
            sparsity_percentage=config.scdv_kwargs['sparsity_percentage'],
            gaussian_mixture_kwargs=dict())
        return scdv

    def _calculate_similarity_train_data(self, click_data, word2items):
        config = WordItemSimilarityConfig()
        matrix_factorization_kwargs = dict(config.matrix_factorization_kwargs)
        matrix_factorization_kwargs['scope_name'] = 'WordItemSimilarityExample'
        matrix_factorization_kwargs['save_directory_path'] = os.path.join(self.local_temporary_directory,
                                                                          'matrix_factorization')
        click_train_data = redshells.app.word_item_similarity.MakeClickTrainData(
            click_data_task=click_data,
            min_user_count=100,
            min_item_count=100,
            max_item_frequency=0.05,
            user_column_name='user_id',
            item_column_name='item_id',
            service_column_name='service_id')
        matrix_factorization = redshells.train.TrainMatrixFactorization(
            train_data_task=click_train_data,
            user_column_name='user_id',
            item_column_name='item_id',
            service_column_name='service_id',
            rating_column_name='click',
            model_kwargs=matrix_factorization_kwargs)
        matrix_factorization_similarity = redshells.app.word_item_similarity.CalculateSimilarityWithMatrixFactorization(
            target_item_task=redshells.data.data_frame_utils.ExtractColumnAsList(
                data_task=click_data, column_name='item_id'),
            matrix_factorization_task=matrix_factorization,
            normalize=True)
        similarity_data = redshells.app.word_item_similarity.MakeSimilarityData(
            word2items_task=word2items,
            similarity_task=matrix_factorization_similarity,
            item_id_0_column_name='item_id_0',
            item_id_1_column_name='item_id_1',
            similarity_column_name='similarity',
            positive_similarity_rate=0.80,
            negative_similarity_rate=0.00,
            output_file_path='word_item_similarity/similarity_data.pkl')
        return similarity_data

    def _calculate_word2items(self, word_data, item_train_data):
        word_matching = redshells.app.word_item_similarity.FindItemKeywordByMatching(
            target_keyword_task=word_data,
            item_task=item_train_data,
            item_id_column_name='item_id',
            item_keyword_column_name='token',
            output_file_path='word_item_similarity/item_keyword.pkl')
        word2items = redshells.data.data_frame_utils.GroupByColumnAsDict(
            data_task=word_matching,
            key_column_name='keyword',
            value_column_name='item_id',
            output_file_path='word_item_similarity/word2items.pkl')
        return word2items

    def _calculate_word2embedding(self, scdv, word_data):
        word2embedding = redshells.app.word_item_similarity.CalculateWordEmbeddingWithSCDV(
            word_task=word_data,
            scdv_task=scdv,
            l2_normalize=True,
            output_file_path='word_item_similarity/word_embedding.pkl')
        return word2embedding

    def _reduce_dimension(self, word2embedding, item2embedding):
        config = WordItemSimilarityConfig()
        dimension_reduction_model = redshells.app.word_item_similarity.TrainDimensionReductionModel(
            item2embedding_task=word2embedding,
            dimension_size=config.similarity_model_kwargs['valid_embedding_size'],
            output_file_path='word_item_similarity/dimension_reduction_model.pkl')
        return redshells.app.word_item_similarity.ApplyDimensionReductionModel(
            item2embedding_task=item2embedding,
            dimension_reduction_model_task=dimension_reduction_model,
            l2_normalize=True,
            output_file_path='word_item_similarity/item2embedding.pkl')

    def _calculate_item2embedding(self, scdv, word2embedding, item_train_data, use_only_title: bool):
        column_name = 'title_token' if use_only_title else 'token'
        item2embedding = redshells.app.word_item_similarity.CalculateDocumentEmbedding(
            document_task=item_train_data,
            scdv_task=scdv,
            item_id_column_name='item_id',
            document_column_name=column_name,
            l2_normalize=True,
            output_file_path='word_item_similarity/document_embedding.pkl')
        item2embedding = self._reduce_dimension(word2embedding, item2embedding)
        return item2embedding

    def _filter_items(self, word2items, word2embedding, item2title_embedding, no_below: float):
        return redshells.app.word_item_similarity.FilterItemByWordSimilarity(
            word2items_task=word2items,
            word2embedding_task=word2embedding,
            item2title_embedding_task=item2title_embedding,
            no_below=no_below)


