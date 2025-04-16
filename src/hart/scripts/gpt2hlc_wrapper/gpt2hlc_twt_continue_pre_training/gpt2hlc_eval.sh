python hartWrapper/src/examples/gpt2hlc_wrapper/run_continue_pt_gpt_twt.py \
    --model_name_or_path hlab/gpt2sml-hlc-twt-v1 \
    --do_eval \
    --output_dir outputs/gpt2_twt_eval \
    --per_device_eval_batch_size 60 \
    --block_size 200 \
    --validation_file $1 \
