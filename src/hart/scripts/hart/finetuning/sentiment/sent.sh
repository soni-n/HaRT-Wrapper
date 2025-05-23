echo $@
CUDA_VISIBLE_DEVICES=$1 \
python -O hartWrapper/src/examples/hart/run_ft_hart.py \
    --learning_rate 7.85427790392214e-06 \
    --early_stopping_patience 6 \
    --model_name_or_path $2 \
    --task_type document \
    --task_name sentiment \
    --num_labels 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
    --metric_for_early_stopping eval_loss \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 15 \
    --per_device_train_batch_size  1 \
    --per_device_eval_batch_size 20 \
    --block_size 1024 \
    --max_train_blocks 8 \
    --output_dir $3 \
    --add_history \
    --initial_history hartWrapper/src/data/initial_history/initialized_history_tensor.pt \
    --train_table hartWrapper/src/data/datasets/sentiment/sent_train_all.pkl \
    --dev_table hartWrapper/src/data/datasets/sentiment/sent_dev_all.pkl \
    --test_table hartWrapper/src/data/datasets/sentiment/sent_test_all.pkl \
    # --overwrite_output_dir \

    
    