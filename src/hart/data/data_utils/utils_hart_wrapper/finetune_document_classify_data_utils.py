import time
import logging
import math
import pandas as pd
from transformers import BatchEncoding

from tqdm import tqdm
tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def format_data(user_id_column, text_id_column, order_by_column, data):
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

# add an argument that decides whether to use the label as the last token or not
def tokenize_with_labels(data, text_column, label_column, tokenizer, use_label_as_insep_token=False):
    def tokenize(data):
        return tokenizer(data)

    def process(data):
        # get the input_ids (i.e., token_ids) and the attention_mask
        # attention_mask is not altered since it's required to attend to all tokens.
        be = tokenize(data[text_column])
        # create the labels of size len(input_ids) and mark all as -100 so that they don't 
        # contribute to the loss calculation
        be['labels'] = [-100] * len(be['input_ids'])
        # except, when the current msg is associated with a label 
        # mark the last token before the separator token as the actual label for stance.
        # this token will be used to predict (i.e., classify into) the label.
        if not math.isnan(data[label_column]) and not use_label_as_insep_token:
            be['labels'][-2] = data[label_column]
        elif not math.isnan(data[label_column]) and use_label_as_insep_token:
            be['input_ids'][-1] = data[label_column]
        return be

    data['tokenized'] = data.apply(process, axis=1)

def normalize_and_concat(data, user_id_column):
    data['tokenized'] = data['tokenized'].apply(lambda x: x.data)
    normalized = pd.json_normalize(data['tokenized'])
    data = pd.concat([data, normalized], axis=1)
    return data.groupby(user_id_column).agg({'input_ids': 'sum', 'attention_mask':'sum', 'labels':'sum'}).reset_index()

def pad_and_chunk(data, tokenizer, block_size):

    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
     
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        l_values = data['labels']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size], labels = l_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        data['input_ids'] = pad(data['input_ids'], tokenizer.eos_token_id)
        data['attention_mask'] = pad(data['attention_mask'], 0)
        data['labels'] = pad(data['labels'], -100)
        return chunks(data)

    data['batch_encodings'] = data.apply(process, axis=1)

def transform_data(logger, tokenizer, data, block_size, text_column, label_column, user_id_column, text_id_column):
    start_time = time.time()
    data_new = data[[user_id_column, text_column, label_column, text_id_column]].copy()
    data_new = data_new.dropna(subset = [text_column])
    append_insep(data_new, tokenizer, text_column)
    tokenize_with_labels(data_new, text_column, label_column, tokenizer)
    data_new = normalize_and_concat(data_new, user_id_column)
    pad_and_chunk(data_new, tokenizer, block_size)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(logger, data, max_blocks):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    return batch.to_numpy().tolist(), actual_blocks

def load_tokenized_dataset(tokenizer, data, block_size=1024, max_blocks=8, text_column=None, label_column=None, user_id_column=None, text_id_column=None, order_by_column=None, retain_original_order=False):
    if isinstance(data, pd.DataFrame):
        data, user_id_column, text_id_column, original_data_order = get_data_from_dataframe(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'pkl' in data:
        data, user_id_column, text_id_column, original_data_order = get_data_from_pkl(logger, data, user_id_column, text_id_column, order_by_column)
    elif 'csv' in data:
        data, user_id_column, text_id_column, original_data_order = get_data_from_csv(logger, data, user_id_column, text_id_column, order_by_column)
    else:
        raise ValueError("Invalid data file format. Please provide a pandas dataframe, or csv, or pkl file")
    
    data = transform_data(logger, tokenizer, data, block_size, text_column, label_column, user_id_column, text_id_column)
    logger.info('************** Block size = {} *************'.format(block_size))
    if retain_original_order:
        return group_data(logger, data, user_id_column, text_id_column, max_blocks), original_data_order
    return group_data(logger, data, max_blocks), None