
export MASTER_PORT=1063

# Check if IN_SERAPH is set to 1
if [ "$IN_SERAPH" -eq 1 ]; then
    echo "IN_SERAPH is set to 1"
    ws=/data/geraugi/plural
    plural=${ws}/../repos/PLURAL
    log_dir=${ws}/logs/finetune
    tensorboard_logdir=${log_dir}/tensorboard_$(date +"%Y-%m-%d_%H-%M-%S")
    checkpoints=${ws}/checkpoints
    save_dir=${checkpoints}/finetune
    bpe_dir=${plural}/utils/BPE
    user_dir=${plural}/ofa_module
    data_dir=${ws}/pre_processed_data
else
    echo "IN_SERAPH is not set to 1 or 0"
    ws=/home/guests/sebastian_sotela
    plural=${ws}/PLURAL
    log_dir=${ws}/logs/reportgen/finetune
    tensorboard_logdir=${log_dir}/tensorboard_$(date +"%Y-%m-%d_%H-%M-%S")
    checkpoints=/home/polyaxon-data/data1/mimic/PLURAL/checkpoints
    save_dir=${checkpoints}/finetune
    bpe_dir=${plural}/utils/BPE
    user_dir=${plural}/ofa_module
    data_dir=${ws}/data_pre_runs/tsv_output_data
fi


mkdir -p $log_dir $save_dir

# Put the larger volume of data in 'data' and the smaller volume of data in 'diff_data'.
# data=${data_dir}/2_samples/mimic_diff_qa_train_2024_11_22_13:01:12.tsv,${data_dir}/2_samples/mimic_diff_qa_valid_2024_11_22_13:01:12.tsv
data=${data_dir}/100_samples/mimic_diff_qa_s2_train.tsv,${data_dir}/100_samples/mimic_diff_qa_s2_valid.tsv

restore_file=${checkpoints}/PLURAL.pt # FIXME: UPDATE
selected_cols=0,1,2,3,4,5,6,7
task=caption
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0
lr=1e-4
max_epoch=8 # 500
warmup_ratio=0 # 0.06
batch_size=8
update_freq=1
resnet_drop_path_rate=0
encoder_drop_path_rate=0
decoder_drop_path_rate=0
dropout=0
weight_decay=0

attention_dropout=0.0
max_src_length=100
max_tgt_length=100
num_bins=1000
patch_image_size=384
eval_cider_cached=${data_dir}/cider_cached_tokens/coco-valid-words.p
drop_worst_ratio=0
drop_worst_after=0
clip_norm=1000


log_file=${log_dir}/$(date +"%Y-%m-%d_%H-%M-%S")_${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}".log"
save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}
mkdir -p $save_path
set -x
ulimit -a
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ${plural}/train.py \
    $data \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=${weight_decay} \
    --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 \
    --clip-norm=${clip_norm} \
    --lr=${lr} \
    --max-epoch=${max_epoch} \
    --log-format=simple \
    --log-interval=10 \
    --fixed-validation-seed=7 \
    --no-epoch-checkpoints --keep-best-checkpoints=1 \
    --save-interval=1 \
    --eval-args='{"beam":5,"max_len_b":16,"no_repeat_ngram_size":3}' \
    --best-checkpoint-metric=loss \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --freeze-encoder-embedding \
    --freeze-decoder-embedding \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --drop-worst-ratio=${drop_worst_ratio} \
    --drop-worst-after=${drop_worst_after} \
    --num-workers=3 \
    --tensorboard-logdir=${tensorboard_logdir}
echo "done"
#    --fp16 \
#    --fp16-scale-window=512 \