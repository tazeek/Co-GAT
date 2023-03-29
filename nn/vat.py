def create_random_tensor():

    ...

def extra_processing():

    ...

def get_original_logits():

    ...

def get_kl_div_loss(original_logits, perturbed_logits):

    ...

def update_gradients_perturbation():

    ...

def perform_vat(model, perturbation_level, utt_list, adj_list, adj_full_list, adj_id_list):

    # Preprocess the data, first and foremost
    var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            model.preprocess_data(utt_list, adj_list, adj_full_list, adj_id_list)
    
    # Get the original logits
    print(var_utt)
    print("\n\n")
    print(var_p)
    print("\n\n")
    print(mask)
    print("\n\n")
    print(len_list)
    print("\n\n")
    exit()

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