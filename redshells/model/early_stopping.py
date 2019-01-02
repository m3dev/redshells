import os
import sys

import tensorflow as tf


class EarlyStopping(object):
    def __init__(self, try_count=1, learning_rate=0., decay_speed=2.0, threshold=0.001, save_directory: str = None):
        self._save_path = os.path.join(save_directory, 'model.ckpt') if save_directory else None
        self._try_count = try_count
        self._leaning_rate = learning_rate
        self._decay_speed = decay_speed
        self._threshold = threshold
        if self._save_path:
            os.makedirs(save_directory, exist_ok=True)
            self._saver = tf.train.Saver()
            self._last_value = sys.float_info.max

    def does_stop(self, value, session: tf.Session) -> bool:
        if self._save_path is None:
            return False

        if self._last_value * (1.0 - self._threshold) < value:
            self._saver.restore(session, self._save_path)
            self._try_count -= 1
            if self._try_count <= 0:
                return True
            self._leaning_rate /= self._decay_speed
            # do not update self._last_value.
            return False

        self._last_value = value
        self._saver.save(session, self._save_path)
        return False

    @property
    def learning_rate(self):
        return self._leaning_rate
