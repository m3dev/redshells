import os
import sys

import tensorflow as tf


class EarlyStopping(object):
    def __init__(self, save_directory: str = None):
        self.save_path = os.path.join(save_directory, 'model.ckpt') if save_directory else None
        if self.save_path:
            self.saver = tf.train.Saver()
            self.last_value = sys.float_info.max

    def does_stop(self, value, session: tf.Session) -> bool:
        if self.save_path is None:
            return False

        if self.last_value < value:
            self.saver.restore(session, self.save_path)
            return True

        self.last_value = value
        self.saver.save(session, self.save_path)
        return False
