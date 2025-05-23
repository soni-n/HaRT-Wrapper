python hartWrapper/src/examples/gpt2hlc_wrapper/run_continue_pt_gpt_twt.py \
    --model_name_or_path hlab/gpt2sml-hlc-twt-v1 \
    --do_train \
    --do_eval \
    --output_dir outputs/gpt2_twt_continue_pt \
    --num_train_epochs 5 \
    --per_device_train_batch_size 60 \
    --per_device_eval_batch_size 60 \
    --block_size 200 \
    --max_train_blocks 8 \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file $1 \
    --validation_file $2 \
