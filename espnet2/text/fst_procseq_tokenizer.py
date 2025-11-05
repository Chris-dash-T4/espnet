import argparse
import h5py
import json
import re
import k2
import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Any

from collections import defaultdict
from tqdm import tqdm

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

from transformers import AutoTokenizer, T5ForConditionalGeneration

class ProcessSequenceTokenizer(AbsTokenizer):

    serialization_dtype = np.dtype([
        ('key1', h5py.string_dtype(encoding='utf-8')),
        ('key2', h5py.string_dtype(encoding='utf-8')),
        ('value', np.float32)
    ])

    def __init__(
        self,
        delimiter: str = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
        **encode_kwargs
    ):
        # **vars(p.parse_known_args("--segmentation-fst fish.fst --idk oth.fst".split(' '))[0])
        assert check_argument_types()
        self.delimiter = delimiter
        self.preprocessor = None
        self.postprocessor = None
        self.punctuation_processor = None

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

        # TODO set a default token list file
        print("Kwargs processed:", encode_kwargs)
        self.tokenlist = encode_kwargs.pop("tokenlist", "tokenlist.txt")
        if self.tokenlist is not None:
            try:
                with open(self.tokenlist, "r", encoding="utf-8") as f:
                    self.tokens = {c.strip(): i+1 for i, c in enumerate(f.readlines())}
                self.tokens["<eps>"] = 0
                self.tokens["#"] = len(self.tokens)
            except FileNotFoundError:
                warnings.warn(f"{self.tokenlist} doesn't exist.")
                self.tokens = {}
        self.rev_tokens = {v:k for k,v in self.tokens.items()}
        self.compute_preprocessor()
        self.compute_punctuation_processor()

        self.fstpath = encode_kwargs.pop("segmentation_fst", None)
        if self.fstpath is not None:
            try:
                with open(self.fstpath, "r", encoding="utf-8") as f:
                    self.fst_base = k2.Fsa.from_openfst(f.read(), acceptor=False)
                    self.fst_base = k2.closure(self.fst_base)
                    self.fst_base = k2.arc_sort(self.fst_base)
            except FileNotFoundError:
                warnings.warn(f"{self.fstpath} doesn't exist.")
                self.fst_base = None
        else:
            self.fst_base = None

        self.preproc_path = encode_kwargs.pop("preproc_fst", None)
        if self.preproc_path is not None:
            try:
                with open(self.preproc_path, "r", encoding="utf-8") as f:
                    self.fst_pre_ling = k2.Fsa.from_openfst(f.read(), acceptor=False)
                    self.fst_pre_ling = k2.arc_sort(self.fst_pre_ling)
            except FileNotFoundError:
                warnings.warn(f"{self.preproc_path} doesn't exist.")
                self.fst_pre_ling = self.univ_acc
        else:
            self.fst_pre_ling = self.univ_acc

        self.lmpath = encode_kwargs.pop("segmentation_lm", None)
        if self.lmpath is not None:
            self.lm = T5ForConditionalGeneration.from_pretrained(self.lmpath, device_map="auto")
            self.lm_tokenizer = AutoTokenizer.from_pretrained(self.lmpath)
        else:
            self.lm = None

        self.beam_size = encode_kwargs.pop("beam_size", 3)

        self.keep_segmentation = encode_kwargs.pop("keep_segmentation", False)
        self.alpha = encode_kwargs.pop("alpha", 0.5)

        self.scores_cache = encode_kwargs.pop("scores_cache", None)
        if self.scores_cache is not None:
            self.scores_cache = Path(self.scores_cache)
            self.scores_for_prefix = defaultdict(lambda: None)
            if self.scores_cache.exists():
                try:
                    with h5py.File(self.scores_cache, "r") as f:
                        ds = f["scores"]
                        self.scores_for_prefix.update({(key1.decode('utf-8'),key2.decode('utf-8')):value for key1,key2,value in ds})
                except:
                    with h5py.File(self.scores_cache, "w") as f:
                        ds = f.create_dataset("scores", shape=(1,), maxshape=(None,),dtype=self.serialization_dtype)
                        ds[0] = np.array([('','',0)],dtype=ds.dtype)
            else:
                self.scores_cache.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(self.scores_cache, "w") as f:
                    ds = f.create_dataset("scores", shape=(1,), maxshape=(None,),dtype=self.serialization_dtype)
                    ds[0] = np.array([('','',0)],dtype=ds.dtype)

        self.seg_cache = encode_kwargs.get("segmentation_cache", None)
        if self.seg_cache is not None:
            if Path(self.seg_cache).exists():
                with open(self.seg_cache, "r", encoding="utf-8") as f:
                    self.seg_cache_local = json.load(f)
            else:
                print(f"File not found: {self.seg_cache}. Creating it...")
                self.seg_cache_local = {}
                try:
                    with open(self.seg_cache, "w", encoding="utf-8") as f:
                        json.dump(self.seg_cache_local,f)
                except:
                    pass

        self.show_lattice_pbar = encode_kwargs.pop("show_lattice_pbar", False)

        self.encode_kwargs = encode_kwargs

    def text2tokens(self, line : str) -> List[str]:
        #print(f'Text: "{line}"')
        if not self.keep_segmentation and (self.seg_cache is None or line.upper() not in self.seg_cache_local):
            # Restricting to word-level segmentation should reduce computation time
            #recog = self.build_word2char_fst(line)
            print("Local not found")
            fst_comp = self.build_acceptor_fst(line)
            fst_comp = k2.compose(fst_comp,self.punctuation_processor)
            fst_comp = k2.compose(fst_comp,self.preprocessor)
            fst_comp = k2.connect(fst_comp)
            fst_comp = k2.compose(fst_comp,self.fst_pre_ling)
            fst_comp = k2.connect(fst_comp)
            # Force the single best decoding after the preprocessor to minimize the number of states
            fst_comp = k2.top_sort(fst_comp)
            fst_comp = k2.one_best_decoding(k2.create_fsa_vec([fst_comp]))[0]
            # Apply the segmentation FST
            fst_comp = k2.compose(fst_comp,self.fst_base)
            # Apply the postprocessor
            fst_comp = k2.compose(fst_comp,self.postprocessor)
            fst_comp = k2.connect(fst_comp)
            fst_comp = k2.remove_epsilon(fst_comp)
            fst_comp = k2.connect(fst_comp)
            line_new = self.find_best_tokenization(fst_comp,line)
            if self.seg_cache is not None:
                self.seg_cache_local[line.upper()] = line_new
                with open(self.seg_cache, "w", encoding="utf-8") as f:
                    json.dump(self.seg_cache_local,f)
            line = line_new
        elif not self.keep_segmentation:
            line = self.seg_cache_local[line.upper()].lower()
        #print(f"Segmentation: {line}")

        # Final step: once segmentation is complete, convert to linearized format
        init = lambda x: re.sub(r"\{([1-4]*)>([1-4]*)\}\{([1-4]+)>([1-4]*)\}",r"{\1\3>\2\4}",x)
        first = lambda x: re.sub(r"^([a-zñ']+)([1-4]*)\{([1-4]*)(>1)?>([1-4]*)\}([1-4]*)(.*)",r'\1\2\3\6\7 \2\3\6\4>\2\5\6',re.sub(r"^([a-zñ']+)([1-4]+)($|[^{1-4].*)",r'\1{\2>\2}\3',x))
        second = lambda x: re.sub(r"^([a-zñ']+[1-4]+)([a-zñ']+)([1-4]*)\{([1-4]*)(>1)?>([1-4]*)\}([1-4]*)(.*)",r'\1\2\3\4\7\8 \3\4\7\5>\3\6\7',re.sub(r"^([a-zñ']+[1-4]+)([a-zñ']+)([1-4]+)($|[^{1-4].*)",r'\1\2{\3>\3}\4',x))
        third = lambda x: re.sub(r"^([a-zñ']+[1-4]+[a-zñ']+[1-4]+)([a-zñ']+)([1-4]*)\{([1-4]*)(>1)?>([1-4]*)\}([1-4]*)(.*)",r'\1\2\3\4\7\8 \3\4\7\5>\3\6\7',re.sub(r"^([a-zñ']+[1-4]+[a-zñ']+[1-4]+)([a-zñ']+)([1-4]+)($|[^{1-4].*)",r'\1\2{\3>\3}\4',x))
        fourth = lambda x: re.sub(r"^([a-zñ']+[1-4]+[a-zñ']+[1-4]+[a-zñ']+[1-4]+)([a-zñ']+)([1-4]*)\{([1-4]*)(>1)?>([1-4]*)\}([1-4]*)(.*)",r'\1\2\3\4\7\8 \3\4\7\5>\3\6\7',re.sub(r"^([a-zñ']+[1-4]+[a-zñ']+[1-4]+[a-zñ']+[1-4]+)([a-zñ']+)([1-4]+)($|[^{1-4].*)",r'\1\2{\3>\3}\4',x))
        linear_all = lambda x: fourth(third(second(first(init(x)))))
        return [token.upper() for word in line.split(' ') for token in linear_all(word).split(' ')]

    def tokens2text(self, tokens : Iterable[str]) -> str:
        res = ' '.join(tokens)
        for match_ in re.finditer(r"=?([A-ZÑÁÉÍÓÚÜ']+[0-9]+)+-?(\s[1-4]+>(1>)?[1-4]+)+",res):
            word = match_.group(0)
            word, *processes = [canonical_forms.get(tok,tok) for tok in word.split(' ')]
            for i,proc in enumerate(processes):
                if proc not in canonical_forms.values():
                    print("Unknown canonical form:",proc)
                prefix = r"[1-4{}>]+[A-ZÑ']+"*i
                word = re.sub(r"(^[A-ZÑ']+"+prefix+r")[1-4]+",lambda m: m.group(1)+proc,word)
            res = re.sub(re.escape(match_.group(0)),word,res)

        if self.keep_segmentation:
            return res
        else:
            res = re.sub(r'\s([!?.,=:])',r'\1',res)
            res = re.sub(r'([¡¿-])\s',r'\1',res)
            res = re.sub(r'\{[^}]*>([^>}]*)}',r'\1',res)
            res = re.sub(r'([1-4])\1',r'\1',res)
            return res

    def build_acceptor_fst(self,text : str) -> k2.Fsa:
        '''Builds an acceptor FST from a text string'''
        tokens_withspace = self.tokens.copy()
        tokens_withspace[' '] = self.tokens['<space>']
        acc_builder = [f"{i} {i+1} {tokens_withspace[c]} {tokens_withspace[c]} 0" for i,c in enumerate(text)]
        # Final state
        acc_builder.append(f"{len(acc_builder)} {len(acc_builder)+1} -1 0 0")
        acc_builder.append(f"{len(acc_builder)}")
        fst_acc = k2.Fsa.from_str('\n'.join(sorted(acc_builder,key=lambda x: int(x.split(' ')[0]))),acceptor=False)
        return fst_acc

    def build_word2char_fst(self,text : str) -> k2.Fsa:
        text_split = {word: i+1 for i,word in enumerate(set(text.split(' ')))}
        acc_builder = [f"1 0 {self.tokens['<eps>']} {self.tokens['<space>']} 0"]
        i_prev = 1
        for word in text_split:
            acc_builder += [f"{i+(i_prev if i > 0 else 0)} {i_prev+i+1} {text_split[word] if i == 0 else self.tokens['<eps>']} {self.tokens[c]} 0" for i,c in enumerate(word)]
            final = acc_builder[-1].split(' ')
            final[1] = "1"
            acc_builder[-1] = ' '.join(final)
            i_prev += len(word)
        # Final state
        q_final = max(map(lambda x: int(x.split(' ')[1]),acc_builder)) + 1
        acc_builder.append(f"0 {q_final} -1 0 0")
        acc_builder.append(f"1 {q_final} -1 0 0")
        acc_builder.append(f"{q_final}")
        fst_acc = k2.Fsa.from_str('\n'.join(sorted(acc_builder,key=lambda x: int(x.split(' ')[0]))),acceptor=False)
        return fst_acc

    def compute_log_probabilities(self, input_text : str, target_text : str, prefix : str = '') -> float:
        '''Computes log probability of target_text given input_text according to the language model'''
        # Update format depending on model
        input_ids = self.lm_tokenizer(f"translate practical to g3: {input_text}", return_tensors="pt").input_ids.to(self.lm.device)
        target_text_full = prefix + target_text
        prefix_len = len(self.lm_tokenizer(text_target=prefix).input_ids[:-1])
        labels = self.lm_tokenizer(text_target=target_text_full, return_tensors="pt").input_ids.to(self.lm.device)

        with torch.no_grad():
            outputs = self.lm(input_ids, labels=labels)
            log_prob = outputs.logits.log_softmax(dim=-1)

        log_prior = log_prob[0,torch.arange(prefix_len,labels.shape[1]-1),labels[0,prefix_len:-1]].sum()
        assert target_text == self.lm_tokenizer.decode(labels[0,prefix_len:-1])

        return log_prior

    def string_from_token_id(self, id : int) -> str:
        if id <= 0:
            return ""
        elif id == self.tokens['<space>']:
            return " "
        else:
            return self.rev_tokens[id].lower()

    def find_best_tokenization(self, fst : k2.Fsa, input_text : str) -> str:
        '''Finds the best tokenization of input_text according to the language model and FST'''
        lm_scores = defaultdict(lambda: float('-inf'))
        prefix_strings = defaultdict(lambda: None)
        if not hasattr(self,"scores_for_prefix"):
            self.scores_for_prefix = defaultdict(lambda: None)
            self.scores_for_prefix[('','')] = 0.0
        scores_to_update = set()

        q0 = 0
        lm_scores[q0] = 0.0
        prefix_strings[q0] = ''

        queue = [(q0,0)]
        arcs = fst.arcs_as_tensor()
        aux : torch.Tensor | k2.ragged.RaggedTensor = fst.aux_labels
        aux_list = aux.tolist()
        final_states = set()
        arc_scores = fst.scores

        pbar = None
        if self.show_lattice_pbar:
            pbar = tqdm(total=len(queue), position=0, leave=None, desc="Lattice traversal progress")
        next_level = {}
        while len(queue) > 0:
            q,level = queue.pop(0)
            prefix = prefix_strings[q]
            if pbar is not None: pbar.update(1)
            outgoing = arcs[:,0] == q

            for arc_idx in torch.where(outgoing)[0]:
                q_next = arcs[arc_idx,1].item()
                token_in = arcs[arc_idx,2].item()
                token_out = aux_list[arc_idx]
                # NB weights are stored as int32, but represent a float
                arc_weight = arc_scores[arc_idx]
                if token_in == -1:
                    final_states.add(q)

                # Can we parallelize this?
                out_new = self.string_from_token_id(token_out) if type(token_out) == int else ''.join([self.string_from_token_id(token) for token in token_out])
                if self.scores_for_prefix[(prefix,out_new)] is None:
                    #print("Cache miss for prefix:", prefix_in_new,", ",prefix_out_new)
                    #print("Queue size:", len(queue))
                    self.scores_for_prefix[(prefix,out_new)] = self.compute_log_probabilities(input_text,out_new,prefix)
                    scores_to_update.add((prefix,out_new))

                lm_score = self.scores_for_prefix[(prefix,out_new)]

                total_logprob = lm_scores[q] + self.alpha * lm_score + (1-self.alpha) * arc_weight
                total_logprob = total_logprob #/ (1 + len(prefix_out))

                #print(token_logprob)
                if total_logprob > lm_scores[q_next]:
                    lm_scores[q_next] = total_logprob
                    prefix_strings[q_next] = prefix+out_new
                    next_level[q_next] = total_logprob

            if len(queue) == 0 and len(next_level) > 0:
                states, probs = tuple(zip(*next_level.items()))
                beam_idxs = torch.topk(torch.tensor(probs),k=min(self.beam_size,len(probs)),sorted=True).indices
                queue += [(states[i.item()],level+1) for i in beam_idxs]
                if pbar is not None: pbar.total += beam_idxs.shape[0]
                next_level = {}

        #best_state = max(final_states, key=lambda x: lm_scores[x])
        q_final = torch.max(fst.arcs_as_tensor()[:,1]).item()

        if self.scores_cache is not None and len(scores_to_update) > 0:
            # Append new entries to cache
            with h5py.File(self.scores_cache,'a') as f:
                ds = f["scores"]
                ds.resize((len(scores_to_update)+ds.shape[0],))
                for i,(prefix,out) in enumerate(scores_to_update):
                    score = self.scores_for_prefix[(prefix,out)].detach().cpu().numpy()
                    new_entry = np.array([(prefix,out,score)],dtype=self.serialization_dtype)
                    ds[-i-1] = new_entry

        return prefix_strings[q_final]


    def compute_preprocessor(self):
        '''Builds a preprocessor (and postprocessor) FST from the token list'''
        if self.preprocessor is None:
            builder = [f"0 1 {self.tokens['<eps>']} {self.tokens['#']} 0"]
            # Main loop: "...word seq..." -> "...word##seq..."
            # map [^#_] to itself before and after the transition
            for token,id in self.tokens.items():
                if token in {'<space>','#','<eps>'}:
                    continue
                builder.append(f"1 1 {id} {id} 0")
            # map _ -> ##
            builder.append(f"1 2 {self.tokens['<space>']} {self.tokens['#']} 0")
            builder.append(f"2 1 {self.tokens['<eps>']} {self.tokens['#']} 0")
            # Final state
            builder.append(f"1 3 {self.tokens['<eps>']} {self.tokens['#']} 0")
            builder.append("3 4 -1 0 0")
            builder.append("4")
            fst_pre = k2.Fsa.from_str('\n'.join(sorted(builder)),acceptor=False)
            fst_pre = k2.arc_sort(fst_pre)
            # Postprocessor is the inverse
            fst_post = k2.invert(fst_pre)
            fst_post = k2.arc_sort(fst_post)
            self.preprocessor = fst_pre
            self.postprocessor = fst_post

            self.univ_acc = k2.Fsa.from_str('\n'.join([f"0 0 {tok} {tok} 0" for tok in self.tokens.values() if tok > 0]) + "\n0 1 -1 0 0\n1",acceptor=False)
            self.univ_acc = k2.arc_sort(self.univ_acc)

    def compute_punctuation_processor(self):
        '''Builds a preprocessor FST to split punctuation from words'''
        if self.punctuation_processor is None:
            builder = [f"0 1 {self.tokens['<eps>']} {self.tokens['<eps>']} 0"]
            builder.append(f"0 2 {self.tokens['<eps>']} {self.tokens['<eps>']} 0")
            builder += [f"1 1 {self.tokens[c]} {self.tokens[c]} 0" for c in self.tokens if c not in (self.non_linguistic_symbols | {'-','='}) and c != '<eps>']
            builder.append(f"1 2 {self.tokens['<eps>']} {self.tokens['<space>']} -1")
            builder.append(f"1 2 {self.tokens['<space>']} {self.tokens['<space>']} 0")
            builder.append(f"1 4 -1 -1 0")
            builder.append(f"1 3 {self.tokens['-']} {self.tokens['-']} 0") # dashes only get final spacing
            builder += [f"2 3 {self.tokens[c]} {self.tokens[c]} 0" for c in self.non_linguistic_symbols if c in self.tokens]
            builder.append(f"2 1 {self.tokens['=']} {self.tokens['=']} 0") # enclitics only get initial spacing
            builder.append(f"3 1 {self.tokens['<eps>']} {self.tokens['<space>']} -1")
            builder.append(f"3 1 {self.tokens['<space>']} {self.tokens['<space>']} 0")
            builder.append(f"3 2 {self.tokens['<eps>']} {self.tokens['<space>']} -0.5")
            builder.append(f"3 2 {self.tokens['<space>']} {self.tokens['<space>']} 0")
            builder.append(f"3 4 -1 -1 0.1")
            builder.append("4")
            fst_punctuation = k2.Fsa.from_str('\n'.join(builder),acceptor=False)
            fst_punctuation = k2.arc_sort(fst_punctuation)
            self.punctuation_processor = fst_punctuation

# Universal canonical forms
trivial = ["1","2","3","4","13","14","21","24","31","32","34","41","42","43","132","134","143","342","412","413","414","421","423","424","431","432","434"]
canonical_forms = {
    "3>4": "{3>4}",
    "1>4": "{1>4}",
	"1>3": "{1>3}",
	"1>4": "{1>4}",
    "1>14": "{1>14}",
	"2>1": "{2>1}",
	"3>1": "{3>1}",
	"3>4": "{3>4}",
	"3>14": "{3>14}",
	"34>14": "{3>14}", # FIXME: This is not the correct canonical form, but FST sometimes generates it
	"3>1>4": "{3>1>4}",
	"3>1>14": "{3>1>14}",
	"1>11": "{>1}1",
	"14>114": "{>1}14",
	"14>4": "{1>}4",
	"3>13": "{>1}3",
	"3>11": "{>1}{3>1}",
	"32>132": "{>1}32",
	"32>42": "{3>4}2",
	"4>14": "{>1}4",
	"4>24": "{>2}4",
	"4>34": "{>3}4",
	"4>44": "{>4}4",
}
for tone in trivial:
    canonical_forms[f"{tone}>{tone}"] = tone

def procseq_kwargs_to_dict(kwargs_as_str : str) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenlist", type=str, required=True)
    parser.add_argument("--segmentation_fst", type=str, required=True)
    parser.add_argument("--preproc_fst", type=str)
    parser.add_argument("--segmentation_lm", type=str)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--keep_segmentation", type=bool, default=False)
    parser.add_argument("--show_lattice_pbar", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for LM weight (0.0: only FST, 1.0: only LM)")
    parser.add_argument("--scores_cache", type=str)
    parser.add_argument("--segmentation_cache", type=str)
    args, unrecognized = parser.parse_known_args(kwargs_as_str.split())
    print("Unknown options:",unrecognized)
    print(f"Parsed args: {args}")
    return vars(args)
