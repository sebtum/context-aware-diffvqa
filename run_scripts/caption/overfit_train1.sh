
export MASTER_PORT=1061

ws=/home/guests/sebastian_sotela
plural=${ws}/PLURAL
log_dir=${ws}/logs/reportgen/pretrain
tensorboard_logdir=${log_dir}/$(date +"%Y-%m-%d_%H-%M-%S")_tensorboard
checkpoints=/home/polyaxon-data/data1/mimic/PLURAL/checkpoints
save_dir=${checkpoints}/pretrain
mkdir -p $log_dir $save_dir

bpe_dir=${plural}/utils/BPE
user_dir=${plural}/ofa_module

data_dir=${ws}/data_pre_runs/tsv_output_data

# Put the larger volume of data in 'data' and the smaller volume of data in 'diff_data'.
# data=${data_dir}/2_samples/mimic_impressions_train_2024_11_22_13:01:12.tsv,${data_dir}/2_samples/mimic_impressions_valid_2024_11_22_13:01:12.tsv
data=${data_dir}/2_samples/mimic_findings_train_2024_11_22_13:01:12.tsv,${data_dir}/2_samples/mimic_findings_valid_2024_11_22_13:01:12.tsv
#data=${data_dir}/mimic_findings_train_20241112_192009.pkl,${data_dir}/mimic_findings_valid_20241112_192009.pkl
#diff_data=${data_dir}/mimic_uni_all_train.tsv,${data_dir}/mimic_uni_all_valid.tsv

restore_file=${checkpoints}/ofa_base.pt
selected_cols=0,1,2,3,4,5,6,7
task=caption
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0
lr=1e-3
max_epoch=10 #500
warmup_ratio=0 #0.06
batch_size=1
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
python3 ${plural}/train.py \
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
    --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08\
    --clip-norm=${clip_norm} \
    --lr=${lr} \
    --max-epoch=${max_epoch} \
    --tensorboard-logdir=${tensorboard_logdir} \
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
    --num-workers=0 > ${log_file} 2>&1