import os
import sys
import json
import torch
import argparse
import time

from utils import DataHub
from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate, vat_training
from utils.dict import PieceAlphabet

def get_hyperparams_args():

    parser = argparse.ArgumentParser()

    # Pre-train Hyper parameter
    parser.add_argument("--pretrained_model", "-pm", type=str, default="none",
                        choices=["none", "bert", "roberta", "xlnet", "albert", "electra"],
                        help="choose pretrained model, default is none.")
    parser.add_argument("--linear_decoder", "-ld", action="store_true", default=False,
                        help="Using Linear decoder to get category.")
    parser.add_argument("--bert_learning_rate", "-blr", type=float, default=1e-5,
                        help="The learning rate of all types of pretrain model.")
    
    # Basic Hyper parameter
    parser.add_argument("--data_dir", "-dd", type=str, default="dataset/mastodon")
    parser.add_argument("--semi_sup_dir","-ssd",type=str, default="dataset/dailydialogue")
    parser.add_argument("--save_dir", "-sd", type=str, default="./save")
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--num_epoch", "-ne", type=int, default=300)
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument("--vat_applied", "-vat", type=bool, default=False)

    # Model Hyper parameter
    parser.add_argument("--num_layer", "-nl", type=int, default=2,
                        help="This parameter CAN NOT be modified! Please use gat_layer to set the layer num of gat")
    parser.add_argument("--gat_layer", "-gl", type=int, default=2,
                        help="Control the number of GAT layers. Must be between 2 and 4.")
    parser.add_argument("--embedding_dim", "-ed", type=int, default=128)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
    parser.add_argument("--gat_dropout_rate", "-gdr", type=float, default=0.1)

    return parser.parse_args()

def build_datasets(data_dir, pretrained_model):

    # Build dataset
    data_house = DataHub.from_dir_addadj(data_dir)

    # piece vocab
    piece_vocab = PieceAlphabet("piece", pretrained_model= pretrained_model)

    return data_house, piece_vocab

def print_trainable_params(model):

    n_trainable_params, n_nontrainable_params = 0, 0

    for p in model.parameters():

        n_params = torch.prod(torch.tensor(p.shape))

        if p.requires_grad:
            n_trainable_params += n_params

        else:

            n_nontrainable_params += n_params

    print('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    return None


args = get_hyperparams_args()
print(json.dumps(args.__dict__, indent=True), end="\n\n\n")

# fix random seed
fix_random_state(args.random_state)

# Get the datasets (labeled and unlabeled)
labeled_data_house, labeled_piece_vocab = build_datasets(args.data_dir, args.pretrained_model)
unlabeled_data_house, unlabeled_piece_vocab = None, None

if args.vat_applied:
    unlabeled_data_house, unlabeled_piece_vocab = build_datasets(args.semi_sup_dir, args.pretrained_model)

model = TaggingAgent(
    labeled_data_house.word_vocab, labeled_piece_vocab, labeled_data_house.sent_vocab,
    labeled_data_house.act_vocab, labeled_data_house.adj_vocab, labeled_data_house.adj_full_vocab, 
    labeled_data_house.adj_id_vocab, args.embedding_dim,
    args.hidden_dim, args.num_layer, args.gat_layer, args.gat_dropout_rate, 
    args.dropout_rate,
    args.linear_decoder, args.pretrained_model
)

if torch.cuda.is_available():
    model = model.cuda()

# For mastodon specific metric
if args.data_dir == "dataset/mastodon":
    use_mastodon_metric = False
else:
    use_mastodon_metric = True

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

print_trainable_params(model)

dev_best_sent, dev_best_act = 0.0, 0.0
test_sent_sent, test_sent_act = 0.0, 0.0
test_act_sent, test_act_act = 0.0, 0.0

start_time = time.time()

for epoch in range(0, args.num_epoch + 1):

    print("Training Epoch: {:4d} ...".format(epoch), file=sys.stderr)

    # Start training
    train_loss, train_time = training(model, labeled_data_house.get_iterator("train", args.batch_size, True),
                                      10.0, args.bert_learning_rate, args.pretrained_model)
    
    # Training dataset update
    print("[Epoch{:4d}], train loss is {:.4f}, cost {:.4f} s.".format(epoch, train_loss, train_time))

    # Perform VAT
    if args.vat_applied:
        vat_loss, vat_time = vat_training(model, labeled_data_house.get_iterator("dev", args.batch_size, True),
                                          10.0, args.bert_learning_rate, args.pretrained_model)

    # Validation dataset
    dev_sent_f1, dev_sent_r, dev_sent_p, dev_act_f1, dev_act_r, dev_act_p, dev_time = evaluate(
        model, labeled_data_house.get_iterator("dev", args.batch_size, False), use_mastodon_metric)
    
    # Testing dataset
    test_sent_f1, sent_r, sent_p, test_act_f1, act_r, act_p, test_time = evaluate(
        model, labeled_data_house.get_iterator("test", args.batch_size, False), use_mastodon_metric)
    
    print("Development Set")
    print("=" * 15)
    print(f"Sentiment:\nF1: {dev_sent_f1}\nRecall: {dev_sent_r}\nPrecision: {dev_sent_p}\n\n")
    print(f"Dialog Act:\nF1: {dev_act_f1}\nRecall: {dev_act_r}\nPrecision: {dev_act_p}\n\n")

    print("Test Set")
    print("=" * 15)
    print(f"Sentiment:\nF1: {test_sent_f1}\nRecall: {sent_r}\nPrecision: {sent_p}\n\n")
    print(f"Dialog Act:\nF1: {test_act_f1}\nRecall: {act_r}\nPrecision: {act_p}\n\n")

    #print("On dev, sentiment f1: {:.4f}, act f1: {:.4f}".format(dev_sent_f1, dev_act_f1))
    #print("On test, sentiment f1: {:.4f}, act f1 {:.4f}".format(test_sent_f1, test_act_f1))
    print("Dev and test cost {:.4f} s.\n".format(dev_time + test_time))

total_training_time = time.time() - start_time
print(f"\n\nTotal training time: {total_training_time:.4f} s.")

torch.save(model, os.path.join(args.save_dir, "model_tazeek_full.pt"))
print("", end="\n")
