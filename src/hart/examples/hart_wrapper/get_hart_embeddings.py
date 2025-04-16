import torch
import pickle
import pandas as pd

from ...model.model_wrapper.hart_model import HaRTModel
from ...data.data_utils.utils_hart_wrapper.inference_data_utils import load_tokenized_dataset as load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from ...data.data_collator import HaRTDefaultDataCollator as hart_default_data_collator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler


def get_hart_embeddings(
        model_path="hlab/hart-gpt2sml-twt-v1",
        data=None,
        text_column=None,
        user_id_column=None,
        text_id_column=None,
        block_size=1024,
        max_blocks=None,
        batch_size=5,
        order_by_column=None,
        retain_original_order=True,
        return_document_embeddings=False,
        return_user_representation_as_mean_user_states=False,
        return_last_token_as_user_representation=False,
        use_insep_as_last_token=False,
        return_all_user_states=False,
        return_output_with_data=False,
        return_word_embeddings=True,
        return_word_embeds_with_insep=False,
        representative_layer='last',
        return_pt = False,
        is_a_test_case=False
        ):
    
    if text_column is None:
        raise ValueError("text_column is required")

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config.pad_token_id = tokenizer.eos_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = HaRTModel(model_path, config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.eval()

    x, original_data_order = load_dataset(
        tokenizer=tokenizer,
        data=data,
        text_column=text_column,
        user_id_column=user_id_column,
        text_id_column=text_id_column,
        block_size=block_size,
        max_blocks=max_blocks,
        order_by_column=order_by_column,
        retain_original_order=retain_original_order
    )
    
    # only for debugging
    # x = x[:35]
    
    user_id_column = user_id_column if user_id_column else 'user_id'
    text_id_column = text_id_column if text_id_column else 'text_id'

    dataloader = DataLoader(
        x,
        sampler=SequentialSampler(x),
        batch_size=batch_size,
        collate_fn=hart_default_data_collator(tokenizer),
    )

    outputs = []

    with torch.no_grad():
        for steps, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) if k in (
                "input_ids", "attention_mask", "labels", "history") else v for k, v in inputs.items()}
            model_outputs = model(**inputs, 
                                  return_all_user_states=return_all_user_states,
                                  return_user_representation_as_mean_user_states=return_user_representation_as_mean_user_states,
                                  return_last_token_as_user_representation=return_last_token_as_user_representation,
                                  return_all_doc_embeds=return_document_embeddings,
                                  return_token_embeds=return_word_embeddings,
                                  representative_layer=representative_layer,
                                  return_token_embeds_with_insep=return_word_embeds_with_insep,
                                  use_insep_as_last_token=use_insep_as_last_token,
                                  )
            outputs.append(model_outputs)

    return_dict = {}
    
    if return_document_embeddings:
        if use_insep_as_last_token:
            doc_embeds_with_text_ids = "doc_embeds_as_insep_with_text_ids"
        else:
            doc_embeds_with_text_ids = "doc_embeds_as_last_token_with_text_ids"

        # combine doc_embeds_with_text_ids into a single dictionary
        doc_embeds_with_text_ids_dict = {}
        for d in outputs:
            for k, v in d[doc_embeds_with_text_ids].items():
                if k not in doc_embeds_with_text_ids_dict:
                    doc_embeds_with_text_ids_dict[k] = []
                doc_embeds_with_text_ids_dict[k].append(v)

        doc_embeds_with_text_ids_df = pd.DataFrame.from_dict(
            doc_embeds_with_text_ids_dict, orient='index').rename(columns={0: 'doc_embeds'})
        
        doc_embeds_with_text_ids_df[text_id_column] = doc_embeds_with_text_ids_df.index
        doc_embeds_with_text_ids_df.reset_index(drop=True, inplace=True)

        if retain_original_order:
            doc_embeds_in_original_order = original_data_order.merge(
                doc_embeds_with_text_ids_df, on=text_id_column, how='left')[[text_id_column, 'doc_embeds']]
            
            #assert no missing values in the merged dataframe and the original data order dataframe 
            assert len(doc_embeds_in_original_order) == len(original_data_order), "Length of doc_embeds_in_original_order and original_data_order should be equal"
            assert doc_embeds_in_original_order[text_id_column].isnull().sum() == 0
            assert original_data_order[text_id_column].isnull().sum() == 0

            return_dict['document_embeddings'] = torch.stack(doc_embeds_in_original_order['doc_embeds'].tolist()) if return_pt else doc_embeds_in_original_order['doc_embeds'].tolist()

            if is_a_test_case:
                return_dict['text_ids'] = doc_embeds_in_original_order[text_id_column].tolist()
        else:
            return_dict['document_embeddings'] = doc_embeds_with_text_ids_df

    if return_user_representation_as_mean_user_states or return_last_token_as_user_representation:
        if return_last_token_as_user_representation:
            if use_insep_as_last_token:
                user_rep = "user_representations_as_last_insep_from"+representative_layer+'layer'
            else:
                user_rep = "user_representations_as_last_token_from"+representative_layer+'layer'
        else:
            user_rep = "user_representations_as_mean_user_states"
        # combine user representations into a single dictionary
        user_reps_with_user_ids = {}
        for d in outputs:
            for k, v in d[user_rep].items():
                if k not in user_reps_with_user_ids:
                    user_reps_with_user_ids[k] = []
                user_reps_with_user_ids[k].append(v)

        user_reps_with_user_ids_df = pd.DataFrame.from_dict(
            user_reps_with_user_ids, orient='index').rename(columns={0: 'user_reps'})
        
        user_reps_with_user_ids_df[user_id_column] = user_reps_with_user_ids_df.index
        user_reps_with_user_ids_df.reset_index(drop=True, inplace=True)

        if retain_original_order:
            user_reps_in_original_order = original_data_order.merge(
                user_reps_with_user_ids_df, on=user_id_column, how='left')[[user_id_column, 'user_reps']]

            # assert no missing values in the merged dataframe and the original data order dataframe
            assert len(user_reps_in_original_order) == len(original_data_order), "Length of user_reps_in_original_order and original_data_order should be equal"
            assert user_reps_in_original_order[user_id_column].isnull().sum() == 0
            assert original_data_order[user_id_column].isnull().sum() == 0
            
            return_dict["user_representations"] = torch.stack(user_reps_in_original_order['user_reps'].tolist()) if return_pt else user_reps_in_original_order['user_reps'].tolist()

            if is_a_test_case:
                return_dict['user_ids'] = user_reps_in_original_order[user_id_column].tolist()
        else:
            return_dict["user_representations"] = user_reps_with_user_ids_df

    if return_all_user_states:
        # combine all_user_states into a single dictionary
        all_user_states_with_user_ids = {}
        for d in outputs:
            for k, v in d['all_user_states'].items():
                if k not in all_user_states_with_user_ids:
                    all_user_states_with_user_ids[k] = []
                all_user_states_with_user_ids[k].append(v[0])

        all_user_states_with_user_ids_df = pd.DataFrame.from_dict(
            all_user_states_with_user_ids, orient='index').rename(columns={0: 'all_user_states'})
        
        all_user_states_with_user_ids_df[user_id_column] = all_user_states_with_user_ids_df.index
        all_user_states_with_user_ids_df.reset_index(drop=True, inplace=True)

        if retain_original_order:
            all_user_states_in_original_order = original_data_order.merge(
                all_user_states_with_user_ids_df, on=user_id_column, how='left')[[user_id_column, 'all_user_states']]
            
            # assert no missing values in the merged dataframe and the original data order dataframe
            assert len(all_user_states_in_original_order) == len(original_data_order), "Length of all_user_states_in_original_order and original_data_order should be equal"
            assert all_user_states_in_original_order[user_id_column].isnull().sum() == 0
            assert original_data_order[user_id_column].isnull().sum() == 0
            
            return_dict['all_user_states'] = torch.stack(all_user_states_in_original_order['all_user_states'].tolist()) if return_pt else all_user_states_in_original_order['all_user_states'].tolist()

            if is_a_test_case:
                return_dict['user_ids'] = all_user_states_in_original_order[user_id_column].tolist()
        else:
            return_dict['all_user_states'] = all_user_states_with_user_ids_df
    
    if return_word_embeddings:
        # combine token_embeds_with_text_ids into a single dictionary
        token_embeds_with_text_ids = {}
        for d in outputs:
            for k, v in d['word_embeds_with_text_ids'].items():
                if k not in token_embeds_with_text_ids:
                    token_embeds_with_text_ids[k] = []
                token_embeds_with_text_ids[k].append(v)

        token_embeds_with_text_ids_df = pd.DataFrame.from_dict(
            token_embeds_with_text_ids, orient='index').rename(columns={0: 'token_ids_and_embeds'})
        
        token_embeds_with_text_ids_df[text_id_column] = token_embeds_with_text_ids_df.index
        token_embeds_with_text_ids_df.reset_index(drop=True, inplace=True)

        # split the token_ids_and_embeds into token_ids and token embeds
        token_embeds_with_text_ids_df[['token_ids', 'token_embeds']] = pd.DataFrame(token_embeds_with_text_ids_df['token_ids_and_embeds'].tolist(), index=token_embeds_with_text_ids_df.index)
        
        if retain_original_order:
            token_ids_embeds_in_original_order = original_data_order.merge(
                token_embeds_with_text_ids_df, on=text_id_column, how='left')
            
            # assert no missing values in the merged dataframe and the original data order dataframe
            assert len(token_ids_embeds_in_original_order) == len(original_data_order), "Length of token_ids_embeds_in_original_order and original_data_order should be equal"
            assert token_ids_embeds_in_original_order[text_id_column].isnull().sum() == 0
            assert original_data_order[text_id_column].isnull().sum() == 0
            
            token_embeds_in_original_order = token_ids_embeds_in_original_order['token_embeds']
            token_ids_in_original_order = token_ids_embeds_in_original_order['token_ids']
            
            return_dict['word_embeddings'] = torch.stack(token_embeds_in_original_order.tolist()) if return_pt else token_embeds_in_original_order.tolist()
            return_dict['token_ids'] = torch.stack(token_ids_in_original_order.tolist()) if return_pt else token_ids_in_original_order.tolist()

            if is_a_test_case:
                return_dict['text_ids'] = token_ids_embeds_in_original_order[text_id_column].tolist()
        else:
            return_dict['word_embeddings'] = token_embeds_with_text_ids_df[[text_id_column, 'token_ids', 'token_embeds']]
                    
    if return_output_with_data:
        data_with_outputs = original_data_order
        assert original_data_order[text_id_column].isnull().sum() == 0
        
        if return_document_embeddings:
            data_with_outputs = data_with_outputs.merge(
                doc_embeds_with_text_ids_df, on=text_id_column, how='left')
            
            #assert no missing values in the merged dataframe and the original data order dataframe
            assert len(data_with_outputs) == len(original_data_order), "Length of data_with_outputs and original_data_order should be equal"
            assert data_with_outputs[text_id_column].isnull().sum() == 0
            
        if return_user_representation_as_mean_user_states:
            data_with_outputs = data_with_outputs.merge(
                user_reps_with_user_ids_df, on=user_id_column, how='left')
            
            #assert no missing values in the merged dataframe and the original data order dataframe
            assert len(data_with_outputs) == len(original_data_order), "Length of data_with_outputs and original_data_order should be equal"
            assert data_with_outputs[user_id_column].isnull().sum() == 0

        if return_all_user_states:
            data_with_outputs = data_with_outputs.merge(
                all_user_states_with_user_ids_df, on=user_id_column, how='left')
            
            #assert no missing values in the merged dataframe and the original data order dataframe
            assert len(data_with_outputs) == len(original_data_order), "Length of data_with_outputs and original_data_order should be equal"
            assert data_with_outputs[user_id_column].isnull().sum() == 0
        
        return_dict['data_with_outputs'] = data_with_outputs
        
    return return_dict
 
#An example run
# outputs = get_hart_embeddings(
#     data='train_data.pkl',
#     text_column='mesage',
#     user_id_column='user_id',
#     text_id_column='message_id',
#     order_by_column='user_msg_order',
#     return_all_user_states=False,
#     return_output_with_data=False,
#     return_word_embeddings=False,
#     return_document_embeddings=True,
#     return_user_representation_as_mean_user_states=False,
# )
