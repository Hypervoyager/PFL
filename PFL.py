import torch
import time
from torch.cuda import random
from logger import Logger
from Node import Node, Global_Node, Select_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Summary
from Trainer import Trainer


# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.split = args.node_num
args.global_model = args.local_model
# args.lr=100
print('Running on', args.device)
Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
Summary(args)

# logs
logger = Logger(args)   


# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]
Select_node = Select_Node(args)

# train
for rounds in range(args.R * args.node_num):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Edge_nodes, args)
    k = Select_node.random_select()
    for epoch in range(args.E):
        Train(Edge_nodes[k])
        recorder.validate(Edge_nodes[k])
    recorder.printer(Edge_nodes[k])
    print('-------------------------')

    Global_node.update(Edge_nodes[k])       # 服务器更新对应的模型参数, 注意，服务器更新的仅是其局部模型，全局模型没更新。这么做是为了避免使用中间变量
    Edge_nodes[k].fork(Global_node)         # 节点从服务器读取全局模型后直接返回
    Global_node.processing()                # 服务器根据其局部模型生成全局模型。可以看到中央服务器的计算过程与边缘节点是可以同时计算的。因此，可以认为是并行计算。

    # log
    recorder.validate(Global_node)
    recorder.printer(Global_node)
    logger.write(rounds=rounds + 1, test_acc=recorder.val_acc[str(Global_node.num)][rounds])

recorder.finish()
logger.close()

Summary(args)

