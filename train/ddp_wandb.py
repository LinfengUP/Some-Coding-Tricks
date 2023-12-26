import argparse
import torch
import os
import wandb
from torch import distributed as dist
import torch.nn as nn
from ddp_checkpoint import save_checkpoint

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)

def get_args():
    parser = argparse.ArgumentParser('Segmentor')
    parser.add_argument('--config', type=str, help='path to config file',default="./pointllm/model/pointbert/experiment-1.yaml")
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--project', type=str, default="pointbert")
    parser.add_argument('--experiment_name', type=str, default="experiment-1")
    args = parser.parse_args()
    return args

def train(args,run=None):

    do_log = run is not None
    is_master = args.local_rank==0

    if args.dist:
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend='nccl')

    model = ToyMpModel().to(rank)
    if args.dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # with random number

    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs,labels).backward()
    optimizer.step()

    # with dataset
    dataset = my_dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.dist else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=20,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )

    for epoch in range(10):
        if args.dist:
            sampler.set_epoch(epoch)
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(rank)
            labels = labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn(outputs, labels).backward()
            optimizer.step()

            if do_log and is_master:
                wandb.log({"loss":loss_fn(outputs, labels).item()})

        if is_master:
            validate(model,epoch,run)
            save_checkpoint(model,epoch,optimizer=optimizer)

def validate(model,epoch,run=None):
    do_log = run is not None
    if do_log:
        run.log({"epoch":epoch})
    pass

if __name__=="__main__":
    args = get_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0 ))
    if args.local_rank==0:
        run = wandb.init(
            project = args.project,
            name = args.experiment_name,
        )

        train(args,run)
    else:
        train(args)