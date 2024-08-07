import torch
import torch.nn.functional as F

def _create_random_tensor(input, xi=1e-6):

    random_noise = torch.rand_like(input)

    # L2 Normalization
    random_noise = xi * F.normalize(random_noise, p=2, dim=-1)
    random_noise.requires_grad_()

    # Mount to CUDA
    if torch.cuda.is_available():
        random_noise = random_noise.cuda()

    return random_noise

def _perturbation_lstm_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, noise = None):

    # Extract the features:
    # - The first is for BiLSTM
    # - The second is for LLMs
    bi_ret = model.extract_utterance_features(var_utt, None)
    #bi_ret = model.extract_utterance_features(var_p, mask)

    # If starting the first time, there won't be noise
    # Hence, generate it
    # Otherwise, skip it
    if noise is None: 
        noise = _create_random_tensor(bi_ret)

    # Add the noise
    perturbed_bi_ret = bi_ret + noise

    # Pass to speaker layer
    perturbed_encoded = model.extract_from_speaker_layer(perturbed_bi_ret, var_adj)

    # Extract from decoder layer via GAT
    sent_h, act_h = model.decode_with_gat(perturbed_encoded, len_list, var_adj_R)

    # Decoding
    pert_pred_sent, pert_pred_act = model(sent_h, act_h)

    # Trim off the fat
    pert_pred_sent, pert_pred_act = _convert_predictions(pert_pred_sent, pert_pred_act, len_list)

    # Return perturbed logits and the perturbation
    return noise, pert_pred_sent

def _perturbation_speaker_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, noise=None):

    # Extract the features:
    # - The first is for BiLSTM
    # - The second is for LLMs
    #bi_ret = model.extract_utterance_features(var_utt, None)
    bi_ret = model.extract_utterance_features(var_p, mask)

    # Pass to speaker layer
    encoded = model.extract_from_speaker_layer(bi_ret, var_adj)

    # If starting the first time, there won't be noise
    # Hence, generate it
    # Otherwise, skip it
    if noise is None: 
        noise = _create_random_tensor(encoded)

    # Add the noise
    perturbed_encoded = encoded + noise

    # Extract from decoder layer via GAT
    sent_h, act_h = model.decode_with_gat(perturbed_encoded, len_list, var_adj_R)

    # Decoding
    pert_pred_sent, pert_pred_act = model(sent_h, act_h)

    # Trim off the fat
    pert_pred_sent, pert_pred_act = _convert_predictions(pert_pred_sent, pert_pred_act, len_list)

    # Return perturbed logits and the perturbation
    return noise, pert_pred_sent, pert_pred_act

def _perturbation_output_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, emo_noise=None, act_noise=None):

    # Extract the features:
    # - The first is for BiLSTM
    # - The second is for LLMs
    #bi_ret = model.extract_utterance_features(var_utt, None)
    bi_ret = model.extract_utterance_features(var_p, mask)

    # Pass to speaker layer
    encoded = model.extract_from_speaker_layer(bi_ret, var_adj)

    # Extract from decoder layer via GAT
    sent_h, act_h = model.decode_with_gat(encoded, len_list, var_adj_R)

    # Add noise to emotion and act
    if emo_noise is None:
        emo_noise = _create_random_tensor(sent_h)

    if act_noise is None:
        act_noise = _create_random_tensor(act_h)

    # Linear Layer
    pert_pred_sent, pert_pred_act = model(sent_h + emo_noise, act_h + act_noise)

    # Trim off the fat
    pert_pred_sent, pert_pred_act = _convert_predictions(pert_pred_sent, pert_pred_act, len_list)

    # Return perturbed logits and the perturbation
    return emo_noise, act_noise, pert_pred_sent, pert_pred_act

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

def _get_original_logits(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R):

    # Extract the features:
    # - The first is for BiLSTM
    # - The second is for LLMs
    bi_ret = model.extract_utterance_features(var_utt, None)
    #bi_ret = model.extract_utterance_features(var_p, mask)

    # Speaker layer next
    full_encoded = model.extract_from_speaker_layer(bi_ret, var_adj)

    # Extract from decoder layer via GAT
    sent_h, act_h = model.decode_with_gat(full_encoded, len_list, var_adj_R)

    # Linear Layer
    pred_sent, pred_act = model(sent_h, act_h)
    #pred_sent = model(sent_h, act_h)

    # Conversion by trimming off the fat off the logits
    #pred_sent = model(sent_h, act_h)
    #pred_act = model(sent_h, act_h)
    pred_sent, _ = _convert_predictions(pred_sent, pred_act, len_list)

    return pred_sent

def _get_kl_div_loss(original_logits, perturbed_logits):

    original = F.softmax(original_logits, dim=-1)
    perturbed = F.log_softmax(perturbed_logits, dim=-1)

    kl_div_loss = F.kl_div(perturbed, original, reduction='batchmean')

    return kl_div_loss

def _update_gradients_perturbation(perturbation, kl_div_loss):

    eps = 1.0

    # Get the updated gradients
    grad, = torch.autograd.grad(kl_div_loss, perturbation)

    # Detach from graph
    perturbed = grad.detach()

    # L2 Normalize and multiply with epsilon
    perturbed = eps * F.normalize(perturbed, p=2, dim=-1)

    return perturbed

def perform_vat(model, perturbation_level, utt_list, adj_list, adj_full_list, adj_id_list):

    # Preprocess the data, first and foremost
    var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            model.preprocess_data(utt_list, adj_list, adj_full_list, adj_id_list)
    
    # Get the original logits
    original_logits_sent, original_logits_act = None, None

    with torch.no_grad():
        #original_logits_sent, original_logits_act = _get_original_logits(
        #    model, var_utt, var_p, mask, var_adj, 
        #    len_list, var_adj_R
        #)

        original_logits_sent = _get_original_logits(
            model, var_utt, var_p, mask, var_adj, 
            len_list, var_adj_R
        )


        #original_logits_act = _get_original_logits(
        #    model, var_utt, var_p, mask, var_adj, 
        #    len_list, var_adj_R
        #)

    # Define the level of perturbation (See Canva document)
    # Perform the necessary preprocessing (as per flow: See Canva document)
    perturbation_raw, pert_logits_sent, pert_logits_act = None, None, None

    perturbation_raw, pert_logits_sent = \
        _perturbation_lstm_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, None)

    #perturbation_raw, pert_logits_sent, pert_logits_act = \
    #    _perturbation_speaker_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, None)

    #perturb_sent, perturb_act, pert_logits_sent, pert_logits_act = \
    #    _perturbation_output_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, None)

    # There are two sets of logits: Sentiment and Act
    # Permutations: Act loss, Emo loss, Dual loss (divide by 2)
    #act_kl_loss = _get_kl_div_loss(original_logits_act, pert_logits_act)
    emo_kl_loss = _get_kl_div_loss(original_logits_sent, pert_logits_sent)

    # Update the gradients of the random tensor, based on the KL Div loss
    # One for Sent, One for Act
    #perturb_act = _update_gradients_perturbation(perturb_act, act_kl_loss)
    #perturb_sent = _update_gradients_perturbation(perturb_sent, emo_kl_loss)
    perturbation_raw = _update_gradients_perturbation(perturbation_raw, emo_kl_loss)

    # Run again with the adjusted perturbation

    _, pert_logits_sent = \
            _perturbation_lstm_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, perturbation_raw)

    #perturbation_raw, pert_logits_sent, pert_logits_act = \
    #    _perturbation_speaker_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, perturbation_raw)

    #perturb_sent, perturb_act, pert_logits_sent, pert_logits_act = \
    #    _perturbation_output_layer(model, var_utt, var_p, mask, var_adj, len_list, var_adj_R, perturb_sent, perturb_act)

    # Get the second KL Div loss (this is based on the updated perturbation)
    #act_kl_loss = _get_kl_div_loss(original_logits_act, pert_logits_act)
    emo_kl_loss = _get_kl_div_loss(original_logits_sent, pert_logits_sent)

    return emo_kl_loss #+ act_kl_loss