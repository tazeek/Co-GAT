import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.encode import BiGraphEncoder
from nn.decode import RelationDecoder, LinearDecoder

from utils.help import ReferMetric
from utils.dict import PieceAlphabet
from utils.load import WordAlphabet, LabelAlphabet
from utils.help import expand_list, noise_augment
from utils.help import nest_list, iterable_support


class TaggingAgent(nn.Module):

    def __init__(self,
                 word_vocab: WordAlphabet,
                 piece_vocab: PieceAlphabet,
                 sent_vocab: LabelAlphabet,
                 act_vocab: LabelAlphabet,
                 adj_vocab: LabelAlphabet,
                 adj_full_vocab: LabelAlphabet,
                 adj_id_vocab: LabelAlphabet,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layer: int,
                 gat_layer: int,
                 gat_dropout_rate: float,
                 dropout_rate: float,
                 use_linear_decoder: bool,
                 pretrained_model: str):

        super(TaggingAgent, self).__init__()

        self._piece_vocab = piece_vocab
        self._pretrained_model = pretrained_model

        self._word_vocab = word_vocab
        self._sent_vocab = sent_vocab
        self._act_vocab = act_vocab
        self._adj_vocab = adj_vocab
        self._adj_full_vocab = adj_full_vocab
        self._adj_id_vocab = adj_id_vocab

        self._encoder = BiGraphEncoder(
            nn.Embedding(len(word_vocab), embedding_dim),
            hidden_dim, dropout_rate, pretrained_model
        )

        if use_linear_decoder:
            self._decoder = LinearDecoder(len(sent_vocab), len(act_vocab), hidden_dim)
        else:
            self._decoder = RelationDecoder(
                len(sent_vocab), len(act_vocab), hidden_dim,
                num_layer, dropout_rate, gat_dropout_rate, gat_layer
            )

        # Loss function
        self._criterion = nn.NLLLoss(reduction="sum")

    # Add for loading best model
    def set_load_best_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)

    def set_load_best_missing_arg_mastodon(self, pretrained_model, layer=2):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)
        self._decoder.add_missing_arg(layer)

    def preprocess_data(self, utt_list, adj_list, adj_full_list, adj_id_list):
        return self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, True)
    
    def decode_with_gat(self, input_h, len_list, adj_re):
        return self._decoder.extract_with_gat(input_h, len_list, adj_re)

    def extract_utterance_features(self, input_w, mask=None):
        return self._encoder.extract_utterances(input_w, mask)
    
    def extract_from_speaker_layer(self, bi_ret, adj):
        return self._encoder(bi_ret, adj)

    def forward(self, sent_h, act_h):
        return self._decoder(sent_h, act_h)

    @property
    def sent_vocab(self):
        return self._sent_vocab

    @property
    def act_vocab(self):
        return self._act_vocab

    def _wrap_padding(self, dial_list, adj_list, adj_full_list, adj_id_list, use_noise):
        
        # Find maximum dialog length
        dial_len_list = [len(d) for d in dial_list]
        max_dial_len = max(dial_len_list)

        # Find the maximum adj len (WHAT IS ADJ?)
        adj_len_list = [len(adj) for adj in adj_list]
        max_adj_len = max(adj_len_list)

        # Find the maximum adj full (What is ADJ FULL?)
        # add adj_full
        adj_full_len_list = [len(adj_full) for adj_full in adj_full_list]
        max_adj_full_len = max(adj_full_len_list)

        # Find the maximum adj full (What is ADJ I?)
        # add adj_I
        adj_id_len_list = [len(adj_I) for adj_I in adj_id_list]
        max_adj_id_len = max(adj_id_len_list)

        # Check via assertion
        assert max_dial_len == max_adj_len, str(max_dial_len) + " " + str(max_adj_len)
        assert max_adj_full_len == max_adj_len, str(max_adj_full_len) + " " + str(max_adj_len)
        assert max_adj_id_len == max_adj_full_len, str(max_adj_id_len) + " " + str(max_adj_full_len)

        # Find the maximum turns in conversation
        turn_len_list = [[len(u) for u in d] for d in dial_list]
        max_turn_len = max(expand_list(turn_len_list))

        # Find the maximum turns in for adj len (WHY?)
        turn_adj_len_list = [[len(u) for u in adj] for adj in adj_list]
        max_turn_adj_len = max(expand_list(turn_adj_len_list))

        # Find the maximum turns in for adj full len (WHY?)
        turn_adj_full_len_list = [[len(u) for u in adj_full] for adj_full in adj_full_list]
        max_turn_adj_full_len = max(expand_list(turn_adj_full_len_list))

        # Find the maximum turns in for adj ID (WHY?)
        turn_adj_id_len_list = [[len(u) for u in adj_I] for adj_I in adj_id_list]
        max_turn_adj_id_len = max(expand_list(turn_adj_id_len_list))

        pad_adj_list = []

        # Perform padding (On Pad Adj)
        for dial_i in range(0, len(adj_list)):
            pad_adj_list.append([])

            for turn in adj_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_len - len(turn))
                pad_adj_list[-1].append(pad_utt)

            if len(adj_list[dial_i]) < max_adj_len:
                pad_dial = [[0] * max_turn_adj_len] * (max_adj_len - len(adj_list[dial_i]))
                pad_adj_list[-1].extend(pad_dial)

        pad_adj_full_list = []

        # Perform padding (On Pad Adj Full)
        for dial_i in range(0, len(adj_full_list)):
            pad_adj_full_list.append([])

            for turn in adj_full_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_full_len - len(turn))
                pad_adj_full_list[-1].append(pad_utt)

            if len(adj_full_list[dial_i]) < max_adj_full_len:
                pad_dial = [[0] * max_turn_adj_full_len] * (max_adj_full_len - len(adj_full_list[dial_i]))
                pad_adj_full_list[-1].extend(pad_dial)

        pad_adj_id_list = []

        # Perform padding (On Pad ID Full)
        for dial_i in range(0, len(adj_id_list)):
            pad_adj_id_list.append([])

            for turn in adj_id_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_id_len - len(turn))
                pad_adj_id_list[-1].append(pad_utt)

            if len(adj_id_list[dial_i]) < max_adj_id_len:
                pad_dial = [[0] * max_turn_adj_id_len] * (max_adj_id_len - len(adj_id_list[dial_i]))
                pad_adj_id_list[-1].extend(pad_dial)

        pad_adj_R_list = []

        # Perform padding (On Pad Adj R)
        for dial_i in range(0, len(pad_adj_id_list)):
            pad_adj_R_list.append([])
            assert len(pad_adj_id_list[dial_i]) == len(pad_adj_full_list[dial_i])
            
            for i in range(len(pad_adj_full_list[dial_i])):
                full = pad_adj_full_list[dial_i][i]
                pad_utt_up = full + full
                pad_adj_R_list[-1].append(pad_utt_up)

            for i in range(len(pad_adj_full_list[dial_i])):
                full = pad_adj_full_list[dial_i][i]
                pad_utt_down = full + full
                pad_adj_R_list[-1].append(pad_utt_down)

        assert len(pad_adj_id_list[0]) * 2 == len(pad_adj_R_list[0]), pad_adj_R_list[0]

        # For adjusting the dialog length
        # and for token conversion (Not Pretrained model)
        pad_w_list, pad_sign = [], self._word_vocab.PAD_SIGN
        #empty_personality_list = [0] * 250

        for dial_i in range(0, len(dial_list)):

            pad_w_list.append([])
            #pad_per_list.append([])

            #conversation_personality = personality_list[dial_i]

            #for personality, turn in zip(conversation_personality, dial_list[dial_i]):
            for turn in dial_list[dial_i]:

                if use_noise:
                    noise_turn = noise_augment(self._word_vocab, turn, 5.0)
                else:
                    noise_turn = turn

                # Tokenization form
                pad_utt = noise_turn + [pad_sign] * (max_turn_len - len(turn))

                # Conversion from tokens to IDs
                pad_w_list[-1].append(iterable_support(self._word_vocab.index, pad_utt))
                #pad_per_list[-1].append(list(personality))

            if len(dial_list[dial_i]) < max_dial_len:

                # Create pads
                pad_dial = [[pad_sign] * max_turn_len] * (max_dial_len - len(dial_list[dial_i]))
                #personality_dial = [empty_personality_list] * (max_dial_len - len(dial_list[dial_i]))
                
                # Add to the list
                pad_w_list[-1].extend(iterable_support(self._word_vocab.index, pad_dial))
                #pad_per_list[-1].extend(personality_dial)

        # For tokenization (Pre-trained models)
        cls_sign = self._piece_vocab.CLS_SIGN
        piece_list, sep_sign = [], self._piece_vocab.SEP_SIGN

        for dial_i in range(0, len(dial_list)):

            piece_list.append([])

            for turn in dial_list[dial_i]:

                seg_list = self._piece_vocab.tokenize(turn)
                piece_list[-1].append([cls_sign] + seg_list + [sep_sign])

            if len(dial_list[dial_i]) < max_dial_len:

                pad_dial = [[cls_sign, sep_sign]] * (max_dial_len - len(dial_list[dial_i]))
                piece_list[-1].extend(pad_dial)

        p_len_list = [[len(u) for u in d] for d in piece_list]
        max_p_len = max(expand_list(p_len_list))

        pad_p_list, mask = [], []


        for dial_i in range(0, len(piece_list)):
            pad_p_list.append([])
            mask.append([])

            for turn in piece_list[dial_i]:
                pad_t = turn + [pad_sign] * (max_p_len - len(turn))
                pad_p_list[-1].append(self._piece_vocab.index(pad_t))
                mask[-1].append([1] * len(turn) + [0] * (max_p_len - len(turn)))
        
        # Convert to Tensors and mount onto Cuda
        var_w_dial = torch.LongTensor(pad_w_list)
        var_p_dial = torch.LongTensor(pad_p_list)
        var_mask = torch.LongTensor(mask)
        var_adj_dial = torch.LongTensor(pad_adj_list)
        var_adj_full_dial = torch.LongTensor(pad_adj_full_list)
        var_adj_R_dial = torch.LongTensor(pad_adj_R_list)

        if torch.cuda.is_available():
            var_w_dial = var_w_dial.cuda()
            var_p_dial = var_p_dial.cuda()
            var_mask = var_mask.cuda()
            var_adj_dial = var_adj_dial.cuda()
            var_adj_full_dial = var_adj_full_dial.cuda()
            var_adj_R_dial = var_adj_R_dial.cuda()

        return var_w_dial, var_p_dial, var_mask, turn_len_list, p_len_list, var_adj_dial, var_adj_full_dial, \
            var_adj_R_dial

    def predict(self, utt_list, adj_list, adj_full_list, adj_id_list):
        
        # Perform preprocessing
        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, False)
        
        # Perform predictions
        bi_ret = None 
        if self._pretrained_model != "none":
            bi_ret = self.extract_utterance_features(var_p, mask)
        else:
            bi_ret = self.extract_utterance_features(var_utt, None)

        # Middle fusion
        full_encoded = self.extract_from_speaker_layer(bi_ret, var_adj)

        # Late fusion
        sent_h, act_h = self.decode_with_gat(full_encoded, len_list, var_adj_R)
        
        # Multitask
        pred_sent, pred_act = self.forward(sent_h, act_h)

        # Single-task
        pred_sent, _ = self.forward(sent_h, act_h)
        #_, pred_act = self.forward(sent_h, act_h)

        # Get the labels
        trim_list = [len(l) for l in len_list]
        
        flat_sent = torch.cat(
            [pred_sent[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )
        
        #flat_act = torch.cat(
        #    [pred_act[i, :trim_list[i], :] for
        #     i in range(0, len(trim_list))], dim=0
        #)

        # Narrow down to top-k (In this case, the top label)
        _, top_sent = flat_sent.topk(1, dim=-1)
        #_, top_act = flat_act.topk(1, dim=-1)

        # Mount to CPU and convert to list
        # Return once done
        sent_list = top_sent.cpu().numpy().flatten().tolist()
        #act_list = top_act.cpu().numpy().flatten().tolist()

        nest_sent = nest_list(sent_list, trim_list)
        #nest_act = nest_list(act_list, trim_list)

        string_sent = iterable_support(
            self._sent_vocab.get, nest_sent
        )
        
        #string_act = iterable_support(
        #    self._act_vocab.get, nest_act
        #)
        
        return string_sent
        #return string_act

        #return string_sent, string_act

    def measure(self, utt_list, sent_list, act_list, adj_list, adj_full_list, adj_id_list):
        
        # Data Preprocessing here

        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, True)

        # Get the gold labels
        flat_sent = iterable_support(
            self._sent_vocab.index, sent_list
        )
        
        flat_act = iterable_support(
            self._act_vocab.index, act_list
        )

        index_sent = expand_list(flat_sent)
        index_act = expand_list(flat_act)

        # Convert to Tensors
        var_sent = torch.LongTensor(index_sent)
        var_act = torch.LongTensor(index_act)

        # Mount to CUDA
        if torch.cuda.is_available():
            var_sent = var_sent.cuda()
            var_act = var_act.cuda()

        # Training starts here
        bi_ret = None 

        if self._pretrained_model != "none":
            bi_ret = self.extract_utterance_features(var_p, mask)
        else:
            bi_ret = self.extract_utterance_features(var_utt, None)

        # Middle fusion - Before the speaker layer
        full_encoded = self.extract_from_speaker_layer(bi_ret, var_adj)

        # Late fusion - Before the GAT layer   
        sent_h, act_h = self.decode_with_gat(full_encoded, len_list, var_adj_R)

        #pred_sent, pred_act = self.forward(sent_h, act_h)
        pred_sent, _ = self.forward(sent_h, act_h)
       
        trim_list = [len(l) for l in len_list]

        # Convert the predictions
        flat_pred_s = torch.cat(
            [pred_sent[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )

        #flat_pred_a = torch.cat(
        #    [pred_act[i, :trim_list[i], :] for
        #     i in range(0, len(trim_list))], dim=0
        #)

        # Calculate the loss after softmax (IMPORTANT)
        sent_loss = self._criterion(
            F.log_softmax(flat_pred_s, dim=-1), var_sent
        )

        #act_loss = self._criterion(
        #    F.log_softmax(flat_pred_a, dim=-1), var_act
        #)

        return sent_loss
        #return act_loss

        #return sent_loss + act_loss
