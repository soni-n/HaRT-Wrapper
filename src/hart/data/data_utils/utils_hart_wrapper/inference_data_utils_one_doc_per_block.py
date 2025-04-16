import time
import logging
import pandas as pd
from transformers import BatchEncoding

from tqdm import tqdm
tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: remove NaNs

def format_data(user_id_column, text_id_column, order_by_column, data):
    original_data_order = data.copy()
    if text_id_column is None:
        data['text_id'] = range(len(data))
        text_id_column = 'text_id'
        original_data_order = data.copy()
    if user_id_column is not None and order_by_column is not None:
        data = data.sort_values(by=[user_id_column, order_by_column])
    if user_id_column is None:
        data['user_id'] = range(len(data))
        user_id_column = 'user_id'
        original_data_order = data.copy()
    if order_by_column is not None:
        data = data.sort_values(by=[order_by_column])
    
    data.reset_index(drop=True, inplace=True)
    return data, user_id_column, text_id_column, original_data_order.reset_index(drop=True)

def get_data_from_csv(logger, csv_file, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(csv_file))
    data = pd.read_csv(csv_file)
    return format_data(user_id_column, text_id_column, order_by_column, data)

def get_data_from_pkl(logger, pkl_file, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from pickle file:{}".format(pkl_file))
    data = pd.read_pickle(pkl_file)
    return format_data(user_id_column, text_id_column, order_by_column, data)

def get_data_from_dataframe(logger, data, user_id_column, text_id_column, order_by_column):
    logger.info("Getting data from dataframe:{}".format(data))
    return format_data(user_id_column, text_id_column, order_by_column, data)

def append_insep(data, tokenizer, text_column):
    data[text_column] = data[text_column] + tokenizer.sep_token

def concat(data, user_id_column, text_column, text_id_column):
    return data.groupby(user_id_column).agg({'batch_encodings': 'sum', text_column: lambda x: ' '.join(x), text_id_column: list}).reset_index()

def process_data(data, tokenizer, text_column, block_size):

    def tokenize(data):
        return tokenizer(data)

    def pad_or_truncate(data, pad_value, last_value):
        if len(data) > block_size:
            data = data[:block_size-1]
            data.append(last_value)
        else:
            multiplier = (block_size - len(data))%block_size
            data.extend([pad_value]*multiplier)
        return data
    
    def process(data):
        tokenized = tokenize(data)
        tokenized['input_ids'] = pad_or_truncate(tokenized['input_ids'], tokenizer.eos_token_id, tokenizer.sep_token_id)
        tokenized['attention_mask'] = pad_or_truncate(tokenized['attention_mask'], 0, 1)
        return [BatchEncoding(dict(input_ids = tokenized['input_ids'], 
                            attention_mask=tokenized['attention_mask']))]

    data['batch_encodings'] = data[text_column].progress_apply(process)

def transform_data(logger, tokenizer, data, block_size, text_column, user_id_column, text_id_column):
    start_time = time.time()
    data_new = data[[user_id_column, text_column, text_id_column]].copy()
    # TODO: Place in correct position
    data_new = data_new.dropna()
    append_insep(data_new, tokenizer, text_column)
    process_data(data_new, tokenizer, text_column, block_size)
    data_new = concat(data_new, user_id_column, text_column, text_id_column)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new

def group_data(logger, data, user_id_column, text_id_column, max_blocks):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info(
        '************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info(
            '************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    assert len(data) == len(batch)
    data = pd.concat((data[[user_id_column, text_id_column]], batch), axis=1)
    return data.to_numpy().tolist()


def load_tokenized_dataset(tokenizer, data, block_size=1024, max_blocks=8, text_column=None, user_id_column=None, text_id_column=None, order_by_column=None, retain_original_order=False):
    if isinstance(data, pd.DataFrame):
        data, user_id_column, text_id_column, original_data_order = get_data_from_dataframe(
            logger, data, user_id_column, text_id_column, order_by_column)
    elif 'pkl' in data:
        data, user_id_column, text_id_column, original_data_order = get_data_from_pkl(
            logger, data, user_id_column, text_id_column, order_by_column)
    elif 'csv' in data:
        data, user_id_column, text_id_column, original_data_order = get_data_from_csv(
            logger, data, user_id_column, text_id_column, order_by_column)
    else:
        raise ValueError(
            "Invalid data file format. Please provide a pandas dataframe, or csv, or pkl file")
    
    ## TODO for debugging
    # data = data[:10]
    
    data = transform_data(logger, tokenizer, data, block_size,
                          text_column, user_id_column, text_id_column)
    logger.info(
        '************** Block size = {} *************'.format(block_size))
    if retain_original_order:
        return group_data(logger, data, user_id_column, text_id_column, max_blocks), original_data_order
    return group_data(logger, data, user_id_column, text_id_column, max_blocks), None
