import time
import logging
import pandas as pd
from transformers import BatchEncoding

from tqdm import tqdm
tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def format_data(user_id_column, text_id_column, order_by_column, label_column, data):
    original_data_order = data.copy()
    if text_id_column is None:
        data['text_id'] = range(len(data))
        text_id_column = 'text_id'
        original_data_order = data.copy()
    if user_id_column is not None and order_by_column is not None:
        data = data.sort_values(by=[user_id_column, order_by_column])
    elif user_id_column is None:
        data['user_id'] = range(len(data))
        user_id_column = 'user_id'
        original_data_order = data.copy()
    elif order_by_column is not None:
        data = data.sort_values(by=[order_by_column])
    
    data.reset_index(drop=True, inplace=True)
    labels = data[[user_id_column, label_column]].copy()
    labels.drop_duplicates(inplace=True)
    return data, user_id_column, text_id_column, labels, original_data_order.reset_index(drop=True)

def get_data_from_csv(logger, csv_file, user_id_column, label_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(csv_file))
    data = pd.read_csv(csv_file)
    return format_data(user_id_column, text_id_column, order_by_column, label_column, data)

def get_data_from_pkl(logger, pkl_file, user_id_column, label_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(pkl_file))
    data = pd.read_pickle(pkl_file)
    return format_data(user_id_column, text_id_column, order_by_column, label_column, data)

def get_data_from_dataframe(logger, data, user_id_column, label_column, text_id_column, order_by_column):
    logger.info("Getting data from dataframe:{}".format(data))
    return format_data(user_id_column, text_id_column, order_by_column, label_column, data)

def append_insep(data, tokenizer, text_column):
    data[text_column] = data[text_column] + tokenizer.sep_token

def concat(data, user_id_column, text_column, text_id_column):
    return data.groupby(user_id_column).agg({text_column: ' '.join, text_id_column: list}).reset_index()

def process_data(data, tokenizer, text_column, block_size):

    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        tokenized = tokenize(data)
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        return chunks(tokenized)

    data['batch_encodings'] = data[text_column].progress_apply(process)

def transform_data(logger, tokenizer, data, block_size, text_column, user_id_column, text_id_column):
    start_time = time.time()
    data_new = data[[user_id_column, text_column, text_id_column]].copy()
    data_new = data_new.dropna(subset = [text_column])
    append_insep(data_new, tokenizer, text_column)
    data_new = concat(data_new, user_id_column, text_column, text_id_column)
    process_data(data_new, tokenizer, text_column, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def join_data_and_labels(data, labels, user_id_column):
    assert len(data)==len(labels)
    merged_data = pd.merge(data, labels, on=user_id_column)
    assert len(merged_data)==len(data)
    assert merged_data.shape[-1]==data.shape[-1]+1
    return merged_data

def group_data(logger, data, user_id_column, text_id_column, label_column, max_blocks):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    assert len(data)==len(batch)
    data = pd.concat((data[[user_id_column, label_column]], batch), axis=1)
    assert data.shape[-1]==batch.shape[-1] + 2
    return data.to_numpy().tolist(), actual_blocks

def load_tokenized_dataset(tokenizer, data, block_size=1024, max_blocks=8, text_column=None, label_column=None, user_id_column=None, text_id_column=None, order_by_column=None, retain_original_order=False):
    if isinstance(data, pd.DataFrame):
        data, user_id_column, text_id_column, labels, original_data_order = get_data_from_dataframe(logger, data, user_id_column, label_column, text_id_column, order_by_column)
    elif 'pkl' in data:
        data, user_id_column, text_id_column, labels, original_data_order = get_data_from_pkl(logger, data, user_id_column, label_column, text_id_column, order_by_column)
    elif 'csv' in data:
        data, user_id_column, text_id_column, labels, original_data_order = get_data_from_csv(logger, data, user_id_column, label_column, text_id_column, order_by_column)
    else:
        raise ValueError("Invalid data file format. Please provide a pandas dataframe, or csv, or pkl file")
    
    data = transform_data(logger, tokenizer, data, block_size, text_column, user_id_column, text_id_column)
    data = join_data_and_labels(data, labels, user_id_column)
    logger.info('************** Block size = {} *************'.format(block_size))
    if retain_original_order:
        return group_data(logger, data, user_id_column, text_id_column, label_column, max_blocks), original_data_order
    return group_data(logger, data, user_id_column, text_id_column, label_column, max_blocks), None