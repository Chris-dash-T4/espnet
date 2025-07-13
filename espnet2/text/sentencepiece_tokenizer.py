from pathlib import Path
from typing import Dict, Iterable, List, Union

import sentencepiece as spm
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class SentencepiecesTokenizer(AbsTokenizer):
    @typechecked
    def __init__(self, model: Union[Path, str], encode_kwargs: Dict = dict()):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None
        self.encode_kwargs = encode_kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line, **self.encode_kwargs)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))

class SentencepieceWrapper(SentencepiecesTokenizer):
    def __init__(self, inner_tokenizer : AbsTokenizer, model: Union[Path, str], encode_kwargs: Dict = dict()):
        super().__init__(model, encode_kwargs)

        self.inner_tokenizer = inner_tokenizer

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        firstpass = self.inner_tokenizer.text2tokens(line)
        return self.sp.EncodeAsPieces(' '.join(firstpass), **self.encode_kwargs)
    
    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.inner_tokenizer.tokens2text(self.sp.DecodePieces(list(tokens)).split(' '))
