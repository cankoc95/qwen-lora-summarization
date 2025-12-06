# DDP training setup for training a Qwen - LoRa adapter on SamSum dataset.
#

from dataset import SamSumDataset, collate_fn

import os
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset, load_from_disk

# Hyperparameters
per_gpu_batch_size = 8 # Use 1 for debugging on mac
grad_accumulation_steps = 8  # Set to 1 to disable it. Helps simulate a larger global batch = (per_gpu_batch * world_size * grad_accumulation_step)
max_seq_len = 1024  # Use 512 for debugging on mac
num_epochs = 10
learning_rate = 1e-5
weight_decay = 0.01
max_grad_norm = 1
eval_interval = 100  # Steps interval for evaluating.

# Inits
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
script_dir = pathlib.Path(__file__).parent.resolve()
data_dir = script_dir / "data"
checkpoint_dir = script_dir / "checkpoints"

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# System settings
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
if torch.cuda.is_available():
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif torch.backends.mps.is_available():
    device = 'mps'
    dtype = 'float32' # or float16
else:
    device = 'cpu'
    dtype = 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ddp = int(os.environ.get("RANK", -1)) != -1 # Are we using ddp?
if ddp:
    # Example 1: Single node, 8 gpus
    # rank between [0, 7]
    # world size 8
    # local rank between [0, 7]
    #
    # Example 2: 2 nodes, 4 gpus each
    # rank between [0, 7]
    # world size 8
    # local rank between [0, 3]
    dist.init_process_group(backend=backend)
    rank = int(os.environ["RANK"])  # Global index of the process [0, world_size - 1]
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of processes/GPUs (N)
    local_rank = int(os.environ["LOCAL_RANK"]) # GPU index local to the node [0, n-1]
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = rank # each process gets a different seed
else:
    master_process = True
    seed_offset = 0

torch.manual_seed(1719 + seed_offset)

@torch.no_grad()
def evaluate(model, val_data_loader, device):
    """ Eval function to evaluate model on the validation dataset"""
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in val_data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item()
        steps += 1

    if ddp:
        # Average all the losses from all gpus/processes
        # Compute total loss and total steps across all gpus. Get average loss by dividing total loss / total steps.
        # This is a better way to get average loss in case data shards wasn't evenly distributed across all gpus.
        loss_tensor = torch.tensor(total_loss, device=device)
        steps_tensor = torch.tensor(steps, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / steps_tensor).item()
    else:
        avg_loss = total_loss / steps

    model.train()
    return avg_loss

def get_scheduler(num_epochs, training_data_loader, optimizer):
    num_training_steps = len(training_data_loader) * num_epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    return get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

def train_one_epoch(model, optimizer, scheduler, train_data_loader, val_data_loader, device, epoch, grad_acc_steps, world_size=None):
    # Put model to training mode
    model.train()
    total_loss = 0.0
    interval_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(train_data_loader):
        # Transfer data to device
        batch = {k: v.to(device) for k, v in batch.items()}
        # Evaluate the loss
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                        )
        loss = outputs.loss
        interval_loss += loss.item() # Unscaled loss. Needed for printing the losses later.
        # Divide the loss by the gradient accumulation steps before calling backward.
        # This is to ensure that we correctly simulate a larger batch size (CE loss does an average over the batch dimension).
        loss = loss / grad_acc_steps
        loss.backward()

        # Gradient accumulation
        if (step + 1) % grad_acc_steps == 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() # Scaled down
        total_steps += 1

        if (step + 1) % eval_interval == 0:
            # Evaluate validation loss
            avg_val_loss = evaluate(model=model, val_data_loader=val_data_loader, device=device)

            # Evaluate train loss across all gpus.
            with torch.no_grad():
                avg_train_loss = torch.tensor(interval_loss / eval_interval, device=device)
                if ddp:
                    dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
                    avg_train_loss = (avg_train_loss / world_size).item()
                else:
                    avg_train_loss = avg_train_loss.item()
                interval_loss = 0.0

            if master_process:
                print(f"Epoch {epoch}, step {step + 1}/{len(train_data_loader)}, train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

    # if ddp:
    #     # Average training loss across all gpus/processes.
    #     avg_loss = torch.tensor(total_loss / max(total_steps, 1), device=device)
    #     dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    #     avg_loss = (avg_loss / world_size).item()
    # else:
    #     avg_loss = total_loss / total_steps

    # if master_process:
    #     print(f"Finished epoch {epoch}, average train loss (across {world_size} GPUs) {avg_loss:.4f}")

def main():
    print(f"Model name {model_name}, device {device} and dtype {ptdtype}")
    print(f"Using ddp: {ddp}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = SamSumDataset(split="train", tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir=data_dir)
    val_ds = SamSumDataset(split="validation", tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir=data_dir)

    pad_token_id = tokenizer.pad_token_id

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

        # Load base model. For DDP, set HF device_map to None
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=ptdtype, device_map=None,)

        train_dl = DataLoader(train_ds, batch_size=per_gpu_batch_size, sampler=train_sampler, collate_fn=lambda b : collate_fn(b, pad_token_id))
        val_dl = DataLoader(val_ds, batch_size=per_gpu_batch_size, sampler=val_sampler, collate_fn=lambda b: collate_fn(b, pad_token_id))
    else:
        train_dl = DataLoader(train_ds, batch_size=per_gpu_batch_size, shuffle=True, collate_fn=lambda b : collate_fn(b, pad_token_id))
        val_dl = DataLoader(val_ds, batch_size=per_gpu_batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id))

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=ptdtype, device_map={"": device},)

    print(f"Size of data in training data loader {len(train_dl)}, validation data loader {len(val_dl)}")

    model.config.use_cache = False

    # Setup model with LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Transfer model to device
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
    scheduler = get_scheduler(num_epochs=num_epochs, training_data_loader=train_dl, optimizer=optimizer)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        if ddp:
            # Makes sure the shuffling is done differently in each epoch.
            train_sampler.set_epoch(epoch)
            train_one_epoch(model=model, optimizer=optimizer, scheduler=scheduler, train_data_loader=train_dl, val_data_loader=val_dl, device=device, epoch=epoch, grad_acc_steps=grad_accumulation_steps, world_size=world_size)
        else:
            train_one_epoch(model=model, optimizer=optimizer, scheduler=scheduler, train_data_loader=train_dl, val_data_loader=val_dl, device=device, epoch=epoch, grad_acc_steps=grad_accumulation_steps)

        if master_process:
            save_dir = f"{checkpoint_dir}/qwen2.5-0.5b-samsum-lora-ddp-epoch{epoch}"
            if ddp:
                # Make sure to unwrap DDP by calling .module
                model.module.save_pretrained(save_dir)
            else:
                model.save_pretrained(save_dir)
            # No change to the tokenizer, but by convention it's simpler to save everything together.
            tokenizer.save_pretrained(save_dir)

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
