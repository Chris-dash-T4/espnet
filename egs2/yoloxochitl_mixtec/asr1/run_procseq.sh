#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
train_dev="dev"
test_set="test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
# tokenlist: newline-separated txt file of FST tokens
# segmentation_fst: path to openfst-formatted segmentation FST
# preproc_fst: path to openfst-formatted preprocessing FST, if any
# segmentation_cache: JSON dictionary mapping unsegmented utterances to previously computed segmentations
# segmentation_lm: (optional if cached) HF-compatible segmentation language model for unrecognized strings
# keep_segmentation: (default: false) whether ASR output is already G3 segmented
procseq_kwargs="--tokenlist ext_models/fst_tokens.txt --segmentation_fst ext_models/fst_segmentation_notminimized.fst --preproc_fst ext_models/preproc.fst --segmentation_cache ext_models/seg_cache.json" #--segmentation_lm ext_models/segment_lm

./asr.sh \
    --local_data_opts "--stage 1" \
    --stage 5 \
    --stop_stage 100 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 5 \
    --inference_nj 5 \
    --use_lm true \
    --token_type procseq \
    --nlsyms_txt "local/nlsyms.txt" \
    --procseq_kwargs "$procseq_kwargs" \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --inference_asr_model valid.acc.best.pth \
    --lm_train_text "data/${train_set}/text"  "$@"

