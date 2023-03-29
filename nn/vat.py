import torch

def _create_random_tensor(input):

    ...

def _perturbation_lstm_layer(model, var_utt, mask, var_adj, len_list, var_adj_R):

    # Extract the features
    bi_ret = model.extract_utterance_features(var_utt, None)

    # Create random tensor
    random_tensor = _create_random_tensor()

    # Add the noise

    # Pass to speaker layer

    # Decoding

    # Trim off the fat

    # Return perturbed logits

    ...

def _convert_predictions(pred_sent, pred_act, len_list):

    # Len list: 2D array
    # Length of inner array: Number of utterances in conversation
    # Value of inner array: Number of tokens in the respective utterance

    # Trim list: Find the number of turns per conversation
    trim_list = [len(l) for l in len_list]

    # Convert the predictions
    # BEFORE TRIMMING: The excess "fat" are the turns not utilized (for padding sake)
    # AFTER TRIMMING: The values that are actually required for classification

    flat_pred_s = torch.cat(
        [pred_sent[i, :trim_list[i], :] for
            i in range(0, len(trim_list))], dim=0
    )

    flat_pred_a = torch.cat(
        [pred_act[i, :trim_list[i], :] for
            i in range(0, len(trim_list))], dim=0
    )

    return flat_pred_s, flat_pred_a

def get_original_logits(model, var_utt, mask, var_adj, len_list, var_adj_R):

    # BiLSTM first
    bi_ret = model.extract_utterance_features(var_utt, None)

    # Speaker layer next
    full_encoded = model.extract_from_speaker_layer(bi_ret, var_adj)

    # The decoding next
    pred_sent, pred_act = model(full_encoded, len_list, var_adj_R)

    # Conversion by trimming off the fat off the logits
    pred_sent, pred_act = _convert_predictions(pred_sent, pred_act, len_list)

    return pred_sent, pred_act

def get_kl_div_loss(original_logits, perturbed_logits):

    ...

def update_gradients_perturbation():

    ...

def perform_vat(model, perturbation_level, utt_list, adj_list, adj_full_list, adj_id_list):

    # Preprocess the data, first and foremost

    var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            model.preprocess_data(utt_list, adj_list, adj_full_list, adj_id_list)
    
    # Get the original logits
    original_logits_sent, original_logits_act = get_original_logits(
        model, var_utt, mask, var_adj, 
        len_list, var_adj_R
    )

    # Define the level of perturbation (See Canva document)
    # Perform the necessary preprocessing (as per flow: See Canva document)
    if perturbation_level == "bilstm_layer":
        _perturbation_lstm_layer(model, var_utt, mask, var_adj, len_list, var_adj_R)

    # Create random tensor and normalize with L2

    # Get the first KL Div loss (this is on the random tensor)

    # Update the gradients of the random tensor, based on the KL Div loss

    # Run again with the adjusted perturbation

    # Get the second KL Div loss (this is based on the updated perturbation)

    # Return the loss (This is the VAT loss)

    ...