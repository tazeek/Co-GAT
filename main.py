import os
import sys
import json
import torch
import argparse
import time
import pickle

#from torch.utils.tensorboard import SummaryWriter

from utils import DataHub
from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate
from utils.process import vat_training, semi_supervised_training
from utils.dict import PieceAlphabet
from pprint import pprint

def get_file_names(args):

    model_file_name = 'model_cogat'
    cm_file_name = 'cm_cogat'
    loss_storage_name = 'loss_tracker'

    # Dataset is the first
    dataset = args.data_dir.split('/')[-1]

    model_file_name += '_' + dataset
    cm_file_name += '_' + dataset
    loss_storage_name += '_' + dataset

    # Followed by VAT
    if args.vat_applied is True:
        model_file_name += '_' + 'vat_' + args.perturbation
        cm_file_name += '_' + 'vat_' + args.perturbation
        loss_storage_name += '_' + 'vat_' + args.perturbation

    return model_file_name, cm_file_name, loss_storage_name

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
    parser.add_argument("--perturbation", "-pert", type=str, default='')

    # Model Hyper parameter
    parser.add_argument("--num_layer", "-nl", type=int, default=2,
                        help="This parameter CAN NOT be modified! Please use gat_layer to set the layer num of gat")
    parser.add_argument("--gat_layer", "-gl", type=int, default=2,
                        help="Control the number of GAT layers. Must be between 2 and 4.")
    parser.add_argument("--embedding_dim", "-ed", type=int, default=128)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
    parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
    parser.add_argument("--gat_dropout_rate", "-gdr", type=float, default=0.1)
    parser.add_argument("--model_name", "-name", type=str, default='')

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

    print(f'Trainable Params:{n_trainable_params}\nNon-trainable Params: {n_nontrainable_params}\n\n')

    return None


args = get_hyperparams_args()
print(json.dumps(args.__dict__, indent=True), end="\n\n\n")

#writer_logs = args.perturbation or 'basic'
#writer = SummaryWriter(f'logs/new_approach_speaker_layer')

# fix random seed
fix_random_state(args.random_state)

# Get the datasets (labeled and unlabeled)
labeled_data_house, labeled_piece_vocab = build_datasets(args.data_dir, args.pretrained_model)
unlabeled_data_house, unlabeled_piece_vocab = None, None

# Get the filenames
model_name, confusion_matrix_name, loss_storage_name = get_file_names(args)

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

loss_storage = {
    'epoch': [],
    'train_loss': [],
    'vat_loss': []
}

for epoch in range(0, args.num_epoch + 1):

    print(f"Training Epoch: {epoch} \n\n")
    train_loss, vat_loss, train_time = 0, 0, None

    # Perform VAT
    if args.vat_applied:

        #vat_loss, vat_time = vat_training(model, labeled_data_house.get_iterator("dev", args.batch_size, True),
        #                                  10.0, args.bert_learning_rate, args.pretrained_model)

        train_loss, vat_loss, train_time = semi_supervised_training(
            model,
            labeled_data_house.get_iterator("train", args.batch_size, True),
            unlabeled_data_house.get_iterator("train", args.batch_size, True),
            10.0, 
            args.bert_learning_rate, 
            args.pretrained_model

        )

        loss_storage['vat_loss'].append(vat_loss)
        #writer.add_scalar('train/vat_loss', vat_loss, epoch)
        print(f"\n[Epoch {epoch} - Training]\nTrain loss is {train_loss:.4f}\nVAT loss is {vat_loss:.4f}\nTime is {train_time:.4f} s.\n\n")
    else:
        # Start training
        train_loss, train_time = training(model, labeled_data_house.get_iterator("train", args.batch_size, True),
                                        10.0, args.bert_learning_rate, args.pretrained_model)
    
    #writer.add_scalar('train/loss', train_loss, epoch)
    
        # Training dataset update
        print(f"\n[Epoch {epoch} - Training]\nTrain loss is {train_loss:.4f}\n\nTime is {train_time:.4f} s.\n\n")

    # Validation dataset (Skip it)
    #dev_sent_f1, dev_sent_r, dev_sent_p, dev_act_f1, dev_act_r, dev_act_p, dev_time = evaluate(
    #    model, labeled_data_house.get_iterator("dev", args.batch_size, False), use_mastodon_metric, None)
    
    loss_storage['epoch'].append(epoch + 1)
    loss_storage['train_loss'].append(train_loss)

    # Testing dataset
    #emo_metrics_output, act_metrics_output, test_time = evaluate(
    #    model, labeled_data_house.get_iterator("test", args.batch_size, False), use_mastodon_metric, confusion_matrix_name)
    
    # Testing dataset - Single (ERC)
    emo_metrics_output, test_time = evaluate(
        model, labeled_data_house.get_iterator("test", args.batch_size, False), use_mastodon_metric, confusion_matrix_name)
    
    # Testing dataset - Single (DARC)
    #act_metrics_output, test_time = evaluate(
    #    model, labeled_data_house.get_iterator("test", args.batch_size, False), use_mastodon_metric, confusion_matrix_name)
    
    #print("Development Set")
    #print("=" * 15)
    #print(f"Sentiment:\nF1: {dev_sent_f1}\nRecall: {dev_sent_r}\nPrecision: {dev_sent_p}\n\n")
    #print(f"Dialog Act:\nF1: {dev_act_f1}\nRecall: {dev_act_r}\nPrecision: {dev_act_p}\n\n")

    #writer.add_scalar('test/emo_f1', test_sent_f1, epoch)
    #writer.add_scalar('test/emo_r', sent_r, epoch)
    #writer.add_scalar('test/emo_p', sent_p, epoch)

    #writer.add_scalar('test/act_f1', test_act_f1, epoch)
    #writer.add_scalar('test/act_r', act_r, epoch)
    #writer.add_scalar('test/act_p', act_p, epoch)

    print("\nTest Set")
    print("=" * 15)
    print("\nEmotion Recognition:\n\n")
    pprint(emo_metrics_output)
    print("=" * 20)
    print("\n")

    #print("=" * 15)
    #print("\nDialog Act Recognition:\n\n")
    #pprint(act_metrics_output)
    #print("=" * 20)
    #print("\n")
    #print(f"Dialog Act Recognition:\n\nF1: {test_act_f1:.4f}\nRecall: {act_r:.4f}\nPrecision: {act_p:.4f}\n\n")

    #print("On dev, sentiment f1: {:.4f}, act f1: {:.4f}".format(dev_sent_f1, dev_act_f1))
    #print("On test, sentiment f1: {:.4f}, act f1 {:.4f}".format(test_sent_f1, test_act_f1))
    #print("Dev and test cost {:.4f} s.\n".format(dev_time + test_time))

total_training_time = time.time() - start_time
print(f"\n\nTotal training time: {total_training_time:.4f} s.")

#torch.save(model, os.path.join(args.save_dir, f"{model_name}.pt"))
#print("", end="\n")

# Save the storage parameters
#with open(f'{loss_storage_name}.pickle', 'wb') as handle:
#        pickle.dump(loss_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
