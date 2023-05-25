import torch
import time
import wandb
from logger import Logger
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Summary
from Trainer import Trainer


# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.split = args.node_num
args.global_model = args.local_model
print('Running on', args.device)
Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
Summary(args)


# logs
wandb.init(project="DFL", entity="paridis")  # 搜一下 wandb的使用方法，entity后填你的用户名
config = wandb.config
config.communications_round = args.R
config.max_lost = args.max_lost
config.node_num = args.node_num
config.dataset = args.dataset
config.local_epoch = args.E
config.optimizer = args.optimizer
config.shuffle = 1
config.local_model = args.local_model
logger = Logger(args)


# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]

# train
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Edge_nodes, args)
    for k in range(len(Edge_nodes)):
        Edge_nodes[k].fork(Global_node)          # download
        for epoch in range(args.E):
            Train(Edge_nodes[k])
            recorder.validate(Edge_nodes[k])
        recorder.printer(Edge_nodes[k])
        print('-------------------------')

    Global_node.merge(Edge_nodes)                # upload
    
    # log
    recorder.validate(Global_node)
    recorder.printer(Global_node)
    logger.write(rounds=rounds + 1, test_acc=recorder.val_acc[str(Global_node.num)][rounds])
    wandb.log({"DFL": recorder.val_acc[str(Global_node.num)][rounds]})
    

recorder.finish()
logger.close()

Summary(args)

