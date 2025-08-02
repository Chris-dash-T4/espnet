import re
import warnings
from pathlib import Path
from typing import Iterable, List, Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


"""
Specialized tokenizer for Yoloxóchitl Mixtec,
separating words into segments and melodies
"""
class SegmentAndMelodyTokenizer(AbsTokenizer):
    def __init__(
        self,
        delimiter: str = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        self.delimiter = delimiter

        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        for t in line.split(self.delimiter):
            if self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols:
                continue

            # Strip prefixes and enclitics
            t = ' ='.join(t.split("="))
            t = '- '.join(t.split("-"))
            # Handle each concatenated morpheme
            prevlen = len(tokens)
            for s in t.split(" "):
                if len(s) == 0: continue
                if s not in self.non_linguistic_symbols:
                    matches = re.finditer(r"([=A-ZÑ']+)([()1-4]+|…)", s)
                    segs = []
                    mels = []
                    for match in matches:
                        segs.append(match.group(1))
                        mels.append(match.group(2) if match.group(2) != '…' else '')
                    if len(segs) == 0:
                        # Spanish word
                        ling = re.search(r"([A-ZÑÁÍÚÉÓÜ']+)", s)
                        if ling is None and s not in self.non_linguistic_symbols:
                            print("ERROR in token", s, f"{bytes(s,'utf-8')}")
                            continue
                        segs.append(ling.group(1))
                    elif s.endswith('-'):
                        segs.append('-')
                        mels.append('')
                    seg_token = "|".join(segs)
                    mel_token = "|".join(mels)

                    tokens.append(seg_token)
                    if mel_token != '': tokens.append(mel_token)

            punctuation = re.findall(f"[{re.escape(''.join(self.non_linguistic_symbols))}]", t)
            if len(punctuation) > 0 and not self.remove_non_linguistic_symbols:
                for p in punctuation:
                    dl = len(tokens) - prevlen
                    if p in list("=¡¿(["):
                        tokens.insert(-dl, p)
                    else:
                        tokens.append(p)

        # TODO sentencepiece???
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter

        words = []
        current_seg = None
        for t in tokens:
            if re.match(r"[A-ZÑ'=-]+\|?", t):
                if current_seg is not None and current_seg[0] not in self.non_linguistic_symbols:
                    words.append(''.join(current_seg))
                elif current_seg is not None:
                    pre = current_seg[0]
                    current_seg = t.split("|")
                    current_seg[0] = pre+current_seg[0]
                    continue
                current_seg = t.split("|")
            elif re.match(r"[()1-4]+\|?", t):
                melody = t.split("|")
                if current_seg is None:
                    print(f"Trailing tone melody: {melody}")
                    continue
                if len(current_seg) != len(melody):
                    print(f"{current_seg} vs {melody}")
                    # Pad invalid tokens
                    melody += ["#"] * len(current_seg)
                    # Crop to match segment length
                    melody = melody[:len(current_seg)]
                word = "".join([s + m for s, m in zip(current_seg, melody)])
                #print(current_seg,"->",word)
                words.append(word)
                current_seg = None
            elif re.match(r"[A-ZÑÁÍÚÉÓÜ]+", t):
                # Spanish word
                if current_seg is not None and current_seg[0] not in self.non_linguistic_symbols:
                    words.append(''.join(current_seg))
                    current_seg = None
                elif current_seg is not None:
                    pre = current_seg[0]
                    words.append(pre+t)
                    current_seg = None
                    continue
                words.append(t)
            elif t in self.non_linguistic_symbols:
                if len(words) > 0 and current_seg is None:
                    words[-1] += t
                elif current_seg is not None:
                    current_seg[-1] += t
                else:
                    current_seg = [t]
            elif t == '':
                continue
            else:
                #raise ValueError(f"invalid token: '{t}' ({len(t)} characters: {[ord(c) for c in t]})")
                print(f"invalid token: '{t}' ({len(t)} characters: {[ord(c) for c in t]})")
                if current_seg is not None and current_seg[0] not in self.non_linguistic_symbols:
                    words.append(''.join(current_seg))
                    current_seg = None
                elif current_seg is not None:
                    pre = current_seg[0]
                    words.append(pre+t)
                    current_seg = None
                    continue
                words.append(t)
        if current_seg is not None:
            words.append(''.join(current_seg))
        return re.sub(r"([(\[¡¿])(\s)",r'\2\1',delimiter.join(words).replace(" =", "=").replace("- ", "-"))
