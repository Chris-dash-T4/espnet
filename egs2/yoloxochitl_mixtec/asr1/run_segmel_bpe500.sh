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

./asr.sh \
    --local_data_opts "--stage 1" \
    --stage 5 \
    --stop_stage 100 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 5 \
    --inference_nj 5 \
    --use_lm true \
    --token_type segmel_bpe \
    --nlsyms_txt "local/nlsyms.txt" \
    --nbpe 500 \
    --bpemode bpe \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --inference_asr_model valid.acc.best.pth \
    --lm_train_text "data/${train_set}/text"  "$@"
