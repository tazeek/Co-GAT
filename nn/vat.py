def create_random_tensor():

    ...

def extra_processing():

    ...

def get_original_logits(model, var_p, mask, var_adj, len_list, var_adj_R):

    # BiLSTM first
    bi_ret = model.extract_utterance_features(var_p, mask)

    # Speaker layer next
    full_encoded = model.extract_from_speaker_layer(bi_ret, var_adj)

    # The decoding next
    pred_sent, pred_act = model(full_encoded, len_list, var_adj_R)

    return None

def get_kl_div_loss(original_logits, perturbed_logits):

    ...

def update_gradients_perturbation():

    ...

def perform_vat(model, perturbation_level, utt_list, adj_list, adj_full_list, adj_id_list):

    # Preprocess the data, first and foremost
    var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            model.preprocess_data(utt_list, adj_list, adj_full_list, adj_id_list)
    
    # Get the original logits

    # Define the level of perturbation (See Canva document)
    # Perform the necessary preprocessing (as per flow: See Canva document)
    if perturbation_level == "bilstm_layer":
        ...

    # Create random tensor and normalize with L2

    # Get the first KL Div loss (this is on the random tensor)

    # Update the gradients of the random tensor, based on the KL Div loss

    # Run again with the adjusted perturbation

    # Get the second KL Div loss (this is based on the updated perturbation)

    # Return the loss (This is the VAT loss)

    ...