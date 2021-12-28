# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import argparse
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--output", default=None, type=str, required=True, help="Output folder where model weights, metrics, preds will be saved")
parser.add_argument("--overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")

#parser.add_argument("--modeltype", default=None, type=str, help="model used [bert ]", required=True)
parser.add_argument("--max_source_text_length", default=512, type=int, help="max length of source text")
parser.add_argument("--max_target_text_length", default=128, type=int, help="max length of target text")
parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

parser.add_argument("--train_batch_size", default=8, type=int, help="Train batch size for training.")
parser.add_argument("--valid_batch_size", default=8, type=int, help="Valid batch size for training.")
#parser.add_argument("-test_batch_size", default=8, type=int, help="Test batch size for evaluation.")
# parser.add_argument("--strategy", default="linear_scheduler" , type=str, help="strategy for finetuning ['linear_scheduler','layerwise_lrd','grouped_layerwise_lrd']")
parser.add_argument("--lr", default=1e-4 , type=float, help="learning rate ")
# parser.add_argument("--head_lr", default=3.6e-6 , type=float, help="learning rate for head")
parser.add_argument("--epochs", default=1, type=int, help="epochs for training")
parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--do_train', type=str, default=False, help="set it to True to train")
parser.add_argument('--do_eval', type=str, default=False, help="set it to True to train")
parser.add_argument('--seed', type=int, default=42, help="set the seed for reproducibility")

args = parser.parse_args()

if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
    raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output))
if not os.path.exists(args.output):
    os.makedirs(args.output)

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(args.seed)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/finetuning_experiments')



class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.source = dataframe.source
        self.target = dataframe.target
        self.source_len = source_len
        self.summ_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):     
        source_text = str(self.source[index])
        target_text = str(self.target[index])
        
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
    
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            
            logger.info(str(epoch), str(_), str(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=128, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%10==0:
                logger.info("Completed {_}")

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def T5Trainer(output_dir=args.output):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    logger.info(f"""[Model]: Loading t5 base...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    # logging
    logger.info(f"[Data]: Reading data...\n")

    train_dataset = pd.read_csv("data/final/train.csv")
    # train_dataset =train_dataset[0:1000]
    logging.info(f"Train dataset shape is {train_dataset.shape}")
    valid_dataset = pd.read_csv("data/final/valid.csv")
    # valid_dataset = valid_dataset[0:100]
    logging.info(f"Valid dataset shape is {valid_dataset.shape}")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(
        train_dataset,
        tokenizer,
        args.max_source_text_length,
        args.max_target_text_length
    )
    val_set = CustomDataset(
        valid_dataset,
        tokenizer,
        args.max_source_text_length,
        args.max_target_text_length
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": args.train_batch_size,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size":  args.valid_batch_size,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr
    )

    # Training loop
    logger.info(f"[Initiating Fine Tuning]...\n")

    for epoch in range(args.epochs):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    logger.info(f"[Saving Model]...\n")
    # Saving the model after training
    #path = os.path.join(output_dir, "model_files")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # evaluating test dataset
    logger.info(f"[Initiating Validation]...\n")
    for epoch in range(args.epochs):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    #console.save_text(os.path.join(output_dir, "logs.txt"))

    logger.info(f"[Validation Completed.]\n")
    logger.info(
        f"""[Model] Model saved @ {output_dir}\n"""
    )
    logger.info(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )


T5Trainer(output_dir=args.output)

