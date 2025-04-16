import torch
from ..hart import HaRTPreTrainedModel
from ..modeling_hart import HaRTBasePreTrainedModel
from ..configuration_hart import HaRTConfig


class HaRTModel(HaRTBasePreTrainedModel):
    def __init__(self, model_name_or_path=None, config=None, pt_model=None, feature_extract_method=None):
        super().__init__(config)
        if model_name_or_path:
            self.transformer = HaRTPreTrainedModel.from_pretrained(
                model_name_or_path, config=config)
        elif pt_model:
            self.transformer = pt_model
        else:
            self.transformer = HaRTPreTrainedModel(config)
            self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.feature_extract_method = feature_extract_method

    def forward(self,
                input_ids=None,
                attention_mask=None,
                history=None,
                output_block_last_hidden_states=True,
                output_block_extract_layer_hs=None,
                user_ids=None,
                text_ids=None,
                return_user_representation_as_mean_user_states=True,
                return_all_user_states=False,
                return_last_token_as_user_representation=False,
                return_all_doc_embeds=True,
                representative_layer='last',
                use_insep_as_last_token=False,
                return_token_embeds=False,
                return_token_embeds_with_insep=False,
                ):
        """
        Forward pass of the HaRT model.

        Args:
                input_ids (torch.Tensor, optional): Input token IDs. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                history (torch.Tensor, optional): User history. Defaults to None.
                output_block_last_hidden_states (bool, optional): Whether to output the last hidden states of the transformer blocks. Defaults to True.
                output_block_extract_layer_hs (bool, optional): Whether to output the hidden states of the specified layer. Defaults to None.
                user_ids (list, optional): List of user IDs. Defaults to None.
                text_ids (list, optional): List of text IDs. Defaults to None.
                return_user_representation_as_mean_user_states (bool, optional): Whether to use mean user states as user representation. Defaults to True.
                return_all_user_states (bool, optional): Whether to return all user states. Defaults to False.
                return_last_token_as_user_representation (bool, optional): Whether to use the last token as user representation. Defaults to False.
                return_all_doc_embeds (bool, optional): Whether to return all document embeddings. Defaults to True.
                representative_layer (str, optional): The layer to extract user and document representations from. Defaults to 'last'. Possible values in ('last', 'second_last')
                use_insep_as_last_token (bool, optional): Whether to use the last token as the last separator token in between user documents. Defaults to False.
                return_token_embeds (bool, optional): Whether to return token embeddings. Defaults to False.

        Returns:
                dict: A dictionary containing the output of the forward pass.
        """

        return_dict = {}

        if history is None:
            raise ValueError(
                "Looks like initial User state was not passed. Please provide U_0.")

        output_doc_embeds_only = input_ids.dim() == 2

        output_block_extract_layer_hs = True if representative_layer == 'second_last' else False

        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)

        messages_transformer_outputs = self.transformer(
            input_ids=input_ids,
            history=history,
            output_block_last_hidden_states=output_block_last_hidden_states,
            output_block_extract_layer_hs=output_block_extract_layer_hs,
            attention_mask=attention_mask,
        )

        if return_user_representation_as_mean_user_states:
            user_representation_as_mean_user_states = self.get_mean_user_states(
                messages_transformer_outputs)
            # assert we have the same number of user representations as user_ids
            assert len(user_representation_as_mean_user_states) == len(
                user_ids)
            # map the user representations to the user_ids
            user_representation_as_mean_user_states = dict(
                zip(user_ids, user_representation_as_mean_user_states))

            user_rep = "user_representations_as_mean_user_states"
            return_dict[user_rep] = user_representation_as_mean_user_states

        if return_last_token_as_user_representation:
            user_representation_as_last_token = self.get_user_rep_as_last_token(
                input_ids,
                messages_transformer_outputs,
                layer=representative_layer,
                last_insep=use_insep_as_last_token
            )
            # assert we have the same number of user representations as user_ids
            assert len(user_representation_as_last_token) == len(user_ids)
            # map the user representations to the user_ids
            user_representation_as_last_token = dict(
                zip(user_ids, user_representation_as_last_token))

            if use_insep_as_last_token:
                user_rep = "user_representations_as_last_insep_from"+representative_layer+'layer'
            else:
                user_rep = "user_representations_as_last_token_from"+representative_layer+'layer'
            return_dict[user_rep] = user_representation_as_last_token

        if return_all_doc_embeds:
            doc_embeds_as_last_token = self.get_document_embeds_as_last_token(
                input_ids,
                messages_transformer_outputs,
                layer=representative_layer,
                last_insep=use_insep_as_last_token
            )
            # assert we have the same number of doc embeddings as text_ids
            assert len(doc_embeds_as_last_token) == len(text_ids)
            # assert we have the same number of doc embeddings as text_ids for each user
            assert all(len(doc_embeds) == len(text_ids[i]) for i, doc_embeds in enumerate(
                doc_embeds_as_last_token))
            # map the doc embeddings to the user_ids and text_ids
            doc_embeds_as_last_token_dict = dict(
                zip(user_ids, zip(text_ids, doc_embeds_as_last_token)))
            # map the doc embeddings to the text_ids only
            # 1. flatten text_ids and doc_embeds_as_last_token
            # 2. zip the flattened text_ids and doc_embeds_as_last_token
            flattened_text_ids = [
                item for sublist in text_ids for item in sublist]
            flattened_doc_embeds_as_last_token = [
                item for sublist in doc_embeds_as_last_token for item in sublist]
            doc_embeds_as_last_token_text_ids = dict(
                zip(flattened_text_ids, flattened_doc_embeds_as_last_token))

            if use_insep_as_last_token:
                doc_embeds = 'document_embeddings_as_insep_from'+representative_layer+'layer'
                doc_embeds_with_text_ids = "doc_embeds_as_insep_with_text_ids"
            else:
                doc_embeds = 'document_embeddings_as_last_token_from'+representative_layer+'layer'
                doc_embeds_with_text_ids = "doc_embeds_as_last_token_with_text_ids"

            return_dict[doc_embeds] = doc_embeds_as_last_token_dict
            return_dict[doc_embeds_with_text_ids] = doc_embeds_as_last_token_text_ids

        if return_all_user_states:
            all_user_states = self.get_all_user_states(
                messages_transformer_outputs)
            # assert we have the same number of user states as user_ids
            assert len(all_user_states) == len(user_ids)
            # map the user states to the user_ids
            all_user_states = dict(zip(user_ids, zip(all_user_states, text_ids)))

            return_dict["all_user_states"] = all_user_states

        if return_token_embeds:
            token_embeds_key = 'word_embeddings_from'+representative_layer+'layer'
            token_embeds_with_textids_key = 'word_embeds_with_text_ids'
            token_embeds, token_ids = self.get_token_embeds_and_ids(
                input_ids, attention_mask, messages_transformer_outputs, layer=representative_layer, with_insep_embeds=return_token_embeds_with_insep)
            # assert we have the same number of sets of set of token embeddings (i.e., users) as text_ids
            assert len(token_embeds) == len(
                token_ids) == len(text_ids) == len(user_ids)
            # assert we have the same number of set of token embeddings (i.e., documents) as text_ids for each user
            assert all(len(documents) == len(
                text_ids[i]) for i, documents in enumerate(token_embeds))
            # map the token embeddings and token ids to the user_ids and text_ids
            token_embeds_dict = dict(
                zip(user_ids, zip(text_ids, token_embeds, token_ids)))
            # map the token embeddings and token ids to the text_ids only
            # 1. flatten text_ids, token_embeds, and token_ids
            # 2. zip the flattened text_ids, token_embeds, and token_ids
            flattened_text_ids = [
                item for sublist in text_ids for item in sublist]
            flattened_token_embeds = [
                item for sublist in token_embeds for item in sublist]
            flattened_token_ids = [
                item for sublist in token_ids for item in sublist]
            # TODO: add assert statement that all flattened lists should have the same length
            token_embeds_text_ids = dict(zip(flattened_text_ids, zip(
                flattened_token_ids, flattened_token_embeds)))

            return_dict[token_embeds_key] = token_embeds_dict
            return_dict[token_embeds_with_textids_key] = token_embeds_text_ids

        return return_dict

    def get_token_pooled_logits_and_ids(self, logits, input_ids, attention_mask, with_insep_embeds=False):
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            raise ValueError(
                "input_ids cannot be None when trying to get doc embeddings for each user.")

        input_ids = input_ids.reshape([batch_size, -1]).cpu()
        logits = logits.reshape([batch_size, -1, logits.shape[-1]]).cpu()
        attention_mask = attention_mask.reshape([batch_size, -1]).to(bool).cpu()

        input_ids_list = []
        logits_list = []

        for i in range(batch_size):

            # get the input_ids and logits corresponding to the attention_mask for each user -- this will remove the padded tokens and logits
            input_ids_list.append(torch.masked_select(input_ids[i], attention_mask[i]).tolist())
            logits_list.append(torch.masked_select(logits[i], attention_mask[i].unsqueeze(-1)).reshape([-1, logits.shape[-1]]))

        assert (
            self.config.sep_token_id is not None
        ), "Cannot handle fetching token embeddings per user if no insep token is defined."

        if self.config.sep_token_id is None:
            raise ValueError(
                "Cannot handle fetching token embeddings per user if no pad token is defined.")
        else:
            pooled_logits = []
            pooled_token_ids = []

            # TODO: Remove
            # # for each user find the index of all the <|insep|> tokens
            # # this corresponds to the indices of the last <|insep|> token for each document for each user
            # indices = [[j for j, x in enumerate(input_ids[i]) if x == self.config.sep_token_id] for i in range(len(input_ids))]

            # run this token embeddings extraction loop for each user
            for i in range(batch_size):
                # TODO: update the comment
                # store the logits corresponding to the found indices in a list
                # this will give the document embeddings using the logits (hidden states)
                # corresponding to the intended token from each document for each user
                start_index = 0
                end_index = 0

                # for each user find the index of all the <|insep|> tokens
                # this corresponds to the indices of the last <|insep|> token for each document for each user
                indices = [j for j, x in enumerate(
                    input_ids_list[i]) if x == self.config.sep_token_id]

                pooled_user_logits = []
                pooled_user_token_ids = []

                # run this token embeddings extraction loop for each document for a given user
                for index in indices:
                    end_index = index

                    if with_insep_embeds:
                        pooled_user_logits.append(
                            logits_list[i][start_index:end_index+1])
                        pooled_user_token_ids.append(
                            input_ids_list[i][start_index:end_index+1])
                    else:
                        pooled_user_logits.append(
                            logits_list[i][start_index:end_index])
                        pooled_user_token_ids.append(
                            input_ids_list[i][start_index:end_index])
                    start_index = end_index+1

                # TODO: check and remove the comment --  handled in forward pass #TODO: fix this code to store a 4 dimensional data : [user, doc, token, embeds]
                # store the logits and corresponding token_ids for a user to the list
                # this will give the document embeddings using the logits (hidden states)
                # corresponding to the intended token from each document for each user
                pooled_logits.append(pooled_user_logits)
                pooled_token_ids.append(pooled_user_token_ids)

        # assert we have the same number of set of documents with token embeddings as set of documents with token_ids for each user
        assert all(len(pooled_logits[i]) == len(pooled_token_ids[i])
                   for i, token_embeds in enumerate(pooled_logits))

        # assert we have the same number of token embeddings as token_ids for each document of each user
        assert all(len(pooled_logits[i][j]) == len(pooled_token_ids[i][j]) for i, documents in enumerate(
            pooled_logits) for j, token_embeds in enumerate(documents))

        return pooled_logits, pooled_token_ids

    def get_token_embeds_and_ids(self, input_ids, attention_mask, messages_transformer_outputs, layer='last', with_insep_embeds=False):
        if layer == 'last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_last_hidden_states
        if layer == 'second_last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_extract_layer_hs
        hidden_states = torch.stack(all_blocks_hidden_states, dim=1).cpu()
        return self.get_token_pooled_logits_and_ids(logits=hidden_states, input_ids=input_ids, attention_mask=attention_mask, with_insep_embeds=with_insep_embeds)

    def get_user_as_last_token_pooled_logits(self, logits, input_ids, inputs_embeds=None, insep=False):

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        input_ids = input_ids.reshape([batch_size, -1])
        logits = logits.reshape([batch_size, -1, logits.shape[-1]])

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                if not insep:
                    # -2 if we want the index of the last token before the <|insep|> token.
                    sequence_lengths = torch.ne(
                        input_ids, self.config.pad_token_id).sum(-1) - 2
                if insep:
                    # -1 if we want the index of the last <|insep|> token.
                    sequence_lengths = torch.ne(
                        input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                self.logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # get the logits corresponding to the indices of the intended tokens (last_token or last <|insep>|) of each user
        # this will give the user representations using the logits (hidden states)
        # corresponding to the intended token from the last document for each user
        pooled_logits = logits[range(batch_size), sequence_lengths]

        return pooled_logits

    def get_user_rep_as_last_token(self, input_ids, messages_transformer_outputs, layer='last', last_insep=False):
        if layer == 'last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_last_hidden_states
        if layer == 'second_last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_extract_layer_hs
        hidden_states = torch.stack(all_blocks_hidden_states, dim=1).cpu()
        input_ids = input_ids.cpu()
        return self.get_user_as_last_token_pooled_logits(hidden_states, input_ids, insep=last_insep)

    def get_document_as_last_token_pooled_logits(self, logits, input_ids, insep=False):

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            raise ValueError(
                "input_ids cannot be None when trying to get doc embeddings for each user.")

        input_ids = input_ids.reshape([batch_size, -1])
        logits = logits.reshape([batch_size, -1, logits.shape[-1]])

        assert (
            self.config.sep_token_id is not None
        ), "Cannot handle fetching doc embeddings per user if no insep token is defined."

        if self.config.sep_token_id is None:
            raise ValueError(
                "Cannot handle fetching doc embeddings per user if no pad token is defined.")
        else:
            # convert input_ids into lists of lists
            input_ids = input_ids.tolist()

            pooled_logits = []

            # run this document embeddings extraction loop for each user
            for i in range(batch_size):
                if insep:
                    # for each user find the index of all the <|insep|> tokens
                    # this corresponds to the indices of the last <|insep|> token for each document for each user
                    indices = [j for j, x in enumerate(
                        input_ids[i]) if x == self.config.sep_token_id]
                if not insep:
                    # for each user find the index of all the tokens right before <|insep|>
                    # this corresponds to the indices of the last token for each document for each user
                    indices = [
                        j-1 for j, x in enumerate(input_ids[i]) if x == self.config.sep_token_id]

                # store the logits corresponding to the found indices in a list
                # this will give the document embeddings using the logits (hidden states)
                # corresponding to the intended token from each document for each user
                pooled_logits.append([logits[i][index] for index in indices])

        return pooled_logits

    def get_document_embeds_as_last_token(self, input_ids, messages_transformer_outputs, layer='last', last_insep=False):
        if layer == 'last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_last_hidden_states
        if layer == 'second_last':
            all_blocks_hidden_states = messages_transformer_outputs.all_blocks_extract_layer_hs
        hidden_states = torch.stack(all_blocks_hidden_states, dim=1).cpu()
        return self.get_document_as_last_token_pooled_logits(hidden_states, input_ids, insep=last_insep)

    def get_masked_all_blocks_user_states_stacked(self, messages_transformer_outputs):
        states = messages_transformer_outputs.history[0]
        masks = messages_transformer_outputs.history[1]
        multiplied = tuple(l * r for l, r in zip(states, masks))
        all_blocks_user_states = torch.stack(multiplied, dim=1).cpu()
        return all_blocks_user_states, masks

    def get_non_pad_block_user_states(self, all_blocks_user_states, masks):
        masks = torch.stack(masks, dim=1).tolist()

        non_pad_block_user_states = []

        for i in range(len(masks)):
            non_pad_block_user_states.append(
                [all_blocks_user_states[i][j] for j in range(len(masks[i])) if masks[i][j][0] == 1])

        return non_pad_block_user_states

    def get_all_user_states(self, messages_transformer_outputs):
        all_blocks_user_states, masks = self.get_masked_all_blocks_user_states_stacked(
            messages_transformer_outputs)
        all_blocks_user_states = self.get_non_pad_block_user_states(
            all_blocks_user_states, masks)
        return all_blocks_user_states

    def get_mean_user_states(self, messages_transformer_outputs):
        all_blocks_user_states, masks = self.get_masked_all_blocks_user_states_stacked(
            messages_transformer_outputs)
        all_blocks_masks = torch.stack(masks, dim=1).cpu()
        sum = torch.sum(all_blocks_user_states, dim=1)
        divisor = torch.sum(all_blocks_masks, dim=1)
        user_representation = sum/divisor
        return user_representation
