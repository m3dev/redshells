from logging import getLogger

from gokart.file_processor import PickleFileProcessor

logger = getLogger(__name__)

try:
    import tensorflow as tf
except ImportError:
    logger.warning('tensorflow is not installed. save_tf_session require tensorflow')


def _get_config(obj):
    var_names = obj.__init__.__code__.co_varnames[:obj.__init__.__code__.co_argcount]
    args = {v: obj.__dict__[v] for v in var_names if v != 'self'}
    return args


def save_tf_session(obj, session: tf.Session, file_path: str):
    tf.train.Saver().save(sess=session, save_path=file_path)
    with open(file_path, 'wb') as f:
        PickleFileProcessor().dump(_get_config(obj), f)


def load_tf_session(cls, session: tf.Session, file_path: str, make_graph):
    with open(file_path, 'rb') as f:
        model = cls(**PickleFileProcessor().load(f))
    model.graph = make_graph(model)
    tf.train.Saver().restore(sess=session, save_path=file_path)
    model.session = session
    return model
