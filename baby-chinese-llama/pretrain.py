import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import time
import math
import pickle
import numpy as numpy
from contextlib import nullcontext
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from model import Transformer, ModelArgs
from dataset import PretrainDataset

def get_logger(filename, verbosity=1, name=None):
    level_dict={0: logging.DEBUG, 1:logging.INFO, 2:logging.WARNING}
    formatter=logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger=logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh=logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh=logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        X=X.to(device)
        Y=Y.to(device)
        lr = get_lr(epoch*iter_per_epoch + step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        if ddp:
            # in ddp training we only need to sync gradients at the last micro step. the official way to do this is with model.no_sync() context manager.
            model.require_bachwark_grad_sync = 0 == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X,Y)
            loss=raw_model.last_loss
            loss=loss/gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1)%gradient_accumulation_steps==0:
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step % log_interval == 0:
                spend_time=time.time()-start_time
                logger.info(
                        'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                            epoch,
                            max_epoch, 
                            step, 
                            iter_per_epoch,
                            loss.item(), 
                            optimizer.param_groups[-1]['lr'],
                            spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        if step % save_interval == 0:
            if ddp:
                if torch.distributed.get_rank() == 0:
                    model.eval()
                    torch.save(model.module.state_dict(),'{}/iter_{}.pth'.format(save_dir,int(step+epoch*iter_per_epoch)))
                    model.train()
            else:
                model.eval()
                torch.save(model.state_dict(),'{}/iter_{}.pth'.format(save_dir,int(step+epoch*iter_per_epoch)))
                model.train()


def init_model():
    model_args=dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,)
    if init_from == "scratch":
        print('Initializing a new model from scratch')
        gptconf = ModelArgs(**model_args)
        model=Transformer(gptconf)
    elif init_from=="resume":
        print(f"Resuming training from {out_dir}")
        ckpt_path=os.path.join(out_dir, 'ckpt.pt')
        checkpoint=torch.load(ckpt_path, map_location=device)
        checkpoint_model_args=checkpoint["model_args"]
        for k in ["dim", "n_layers" "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k]=checkpoint_model_args[k]
    return model

        



if __name__ == "__main__":
    out_dir='/data/xfli/ckpts/babyllama2/'
    max_epoch=1
    eval_interval=1
    log_interval=1
    save_interval=10000
    eval_iters=200
    eval_only=False
    always_save_checkpoint=False
    init_from="scratch"
    gradient_accumulation_steps=1
    batch_size=32
    # model
    max_seq_len=512
    dim=512
    n_layers=8
    n_heads=8
    multiple_of=32
    dropout=0.0
    bias=False
    # adaw optimizer
    learning_rate=3e-4
    weight_decay=1e-1
    beta1=0.9
    beta2=0.95
    grad_clip=1.0
    decay_lr=True
    warmup_iters=1000
    lr_decay_iters=80000
    min_lr=1e-5
    # DPP setting
    backend='nccl' # 'nccl'
    # system
    device='cuda'
    dtype='float16'
    compile=False # use PyTorch 2.0 to compile model to be faster
    config_keys=[
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    save_dir=os.path.join(out_dir, 'pretrain')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    logger = get_logger(os.path.join(save_dir, 'log.log'))

    ddp=int(os.environ.get("RANK", -1))!=-1 # is this a ddp run?
    print(f"use ddp {ddp}")
    # 配置多卡训练
    if ddp:
        if os.name=='nt':
            init_process_group(backend='gloo')
        else:
            init_process_group(backend='nccl')
        ddp_rank=int(os.environ['RANK'])
        ddp_local_rank=int(os.environ["LOCAL_RANK"])
        ddp_world_size=int(os.environ["WORLD_SIZE"])
        device=f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process=ddp_rank==0
        seed_offset=ddp_rank    # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        #assert gradient_accumulation_steps % ddp_world_size == 0
        #gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process=True
        seed_offset=0
        ddp_world_size=1
    token_per_iter=gradient_accumulation_steps*ddp_world_size*batch_size*max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {token_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)    
    
    # TODO 
    torch.manual_seed(1337+seed_offset)
    torch.backends.cuda.matmul_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device_type='cuda' if 'cuda' in device else 'cpu'
    ptdtype={"float32":torch.float32, "bfloat16":torch.bfloat16, "float16":torch.float16}[dtype]
    ctx=(nullcontext() if device_type=='cpu' else torch.cuda.amp.autocast())   

    best_val_loss=1e9

    # init dataloader
    data_path='/data/xfli/data/tokenized/'
    data_path_list=[os.path.join(data_path, x) for x in os.listdir(data_path)]
    data_path_list=['/data/xfli/data/tokenized/wiki.bin']

    train_ds=PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_ds)
    # TODO 这个dataloader不需要collate fn
    train_loader=torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0 if os.name=='nt' else 4,
        sampler=train_sampler
    )
    model=init_model()
    model.to(device)
    # init a grad scaler
    scaler=torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))
    # optimzier 
    optimizer=model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2), device_type)
    # compile the model TODO
    if compile:
        print("compiling the model...")
        unoptimized_model=model
        model=torch.compile(model)
    if ddp:
        prefix='_orig_mod.' if compile else ""
        model._ddp_params_and_buffers_to_ignore={prefix+"freqs_cis"}    # 这一部分不用ddp
        model=DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model  # unwrap ddp container if needed
    # training loop
    iter_per_epoch=len(train_loader)
    for epoch in range(max_epoch):
        train_epoch(epoch)
        if ddp:
            if torch.distributed.get_rank()==0:
                torch.save(raw_model.state_dict(), "{}/epoch_{}".format(save_dir, epoch))
        else:
            torch.save(raw_model.state_dict(), "{}/epoch_{}".format(save_dir, epoch))
    if ddp:
        destroy_process_group()
    