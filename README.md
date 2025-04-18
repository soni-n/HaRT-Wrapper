# HaRT-Wrapper

Code in this repository is a wrapper of the original HaRT model's code to enable user-friendly installation and APIs. <br/>
See [Original HaRT repository](https://github.com/humanlab/HaRT) based on the paper [Human Language Modeling](https://aclanthology.org/2022.findings-acl.52/)

# Setup

### Requires Python 3.x (tested with Python 3.8)
```
pip install hart-wrapper
```

# Fetch embeddings

## Default Examples for Word Embeddings
```
from hart import get_hart_embeddings

#An example run when the data does not have any user information (e.g., user_id, text_id, ordering)
outputs = get_hart_embeddings(
     data='data.pkl',
     text_column='text',
)

#An example run when the data does has user information (e.g., user_id, text_id, ordering)
outputs = get_hart_embeddings(
     data='train_data.pkl',
     text_column='text',
     user_id_column='user_id',
     text_id_column='text_id',
     order_by_column='user_text_order'
)

Note: If one or more of user_id_column, text_id_column, and order_by_column is not given, the code handles it internally. However, if you have temporal/ordering information, you should pass it for better results.
```

## More options available for Document Embeddings, User Representations, and User States

```
The following arguments are available to use:

model_path="hlab/hart-gpt2sml-twt-v1",                # Defaults to publicly available HaRT model. Can use custom trained/fine-tuned HaRT model paths.
batch_size=5,
retain_original_order=True,
return_document_embeddings=False,                     # Returns document representations (embeds).
return_user_representation_as_mean_user_states=True,  # Whether to use mean user states as user representation. Defaults to True.
return_last_token_as_user_representation=False,       # Whether to use the last token as user representation. Defaults to False.
use_insep_as_last_token=False,                        # Whether to use the last token as the last separator token in between user documents. Defaults to False.
return_all_user_states=False,                         # Whether to return all user states (from all blocks). Defaults to False.
return_output_with_data=False,                        # if retain_original_order is set to False, this can be set to true to return the data inputs with correspnding embeds.
return_word_embeddings=True,
return_word_embeds_with_insep=False,                  # returns embeds for <|insep|> special token as well.
representative_layer='last',                          # The layer to extract words, user, and document representations from. Defaults to 'last'. Possible values in ('last', 'second_last')
return_pt = False,                                    # return embeds as PyTorch Tensors or as lists.
```

# Cite 

```
@inproceedings{soni2022human,
  title={Human Language Modeling},
  author={Soni, Nikita and Matero, Matthew and Balasubramanian, Niranjan and Schwartz, H Andrew},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={622--636},
  year={2022}
}
```

