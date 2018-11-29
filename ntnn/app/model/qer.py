import asyncio
from konlpy.tag import Mecab
import numpy as np
import os
from tensorflow.contrib.predictor import from_saved_model

from ntnn.nn.qer.idmapper import IdMapper
from ntnn.nn.qer.predict_util import select_entities
from ntnn.nn.qer.common import Config
from ntnn.nn.qer.common import sentence_to_id, tags_to_id, clear_str, morps


class QER:
    def __init__(self, model_dir):
        self.config = Config()
        self.mecab = Mecab()
        self.mapper = IdMapper(
            os.path.join(model_dir, 'tag.pkl'),
            vocab_size=self.config.max_tags)
        self.predictor = from_saved_model(export_dir=model_dir)

    async def parse(self, text):
        maxwords = self.config.max_words_per_sentence
        maxchars = self.config.max_chars_per_word
        sentences, tags = morps(text, self.mecab, maxwords)

        lens = [min(len(st), maxwords) for st in sentences]
        sentences_ = [[clear_str(morp) for morp in st] for st in sentences]
        char_ids = [
            sentence_to_id(st, maxwords, maxchars) for st in sentences_]
        tags = [tags_to_id(st, self.mapper, maxwords) for st in tags]

        return {
            'x': np.array(char_ids, dtype=np.int32),
            'tags': np.array(tags, dtype=np.int32),
            'seq_lens': np.array(lens, dtype=np.int32)
        }, sentences

    async def predict(self, text):
        features, sentences = await asyncio.ensure_future(self.parse(text))

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, self.predictor, features)
        return select_entities(res['class'], sentences)
