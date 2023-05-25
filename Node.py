import copy
from re import S
from numpy import s_
import torch
# from torch.cuda import random
import random
import Model


def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet18':
        model = Model.ResNet18()
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data

        self.model = init_model(self.args.local_model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)

        # self.global_model = init_model(self.args.global_model).to(self.device)
        # self.global_optimizer = init_optimizer(self.global_model, self.args)

        ## 
        # self.s_list = []   # 存储待选择的节点名单
        # self.c_list = []   # 已选择的节点名单
        # self.node_list = list(range(1, args.node_num))
        # self.max_lost = 1   # 最快节点与最慢节点的通讯回合差

        # for j in range(self.max_lost):
        #     self.s_list.extend(self.node_list)

    def fork(self, global_node):
        # 每次迭代后，接收新的global_model
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)


    # def random_select(self):
    #     index = random.randrange(len(self.s_list))      # 随机取下标
    #     chosen_number = self.s_list.pop(index)          # 找到下标对应的节点序号
    #     self.c_list.append(chosen_number)               # 中间变量，衡量是否全部节点完成一次通讯
    #     print(self.c_list)

    #     if len(set(self.c_list)) == self.args.node_num :
    #         self.s_list.extend(self.node_list)          # 判断为真，代表所有节点完成一次迭代，此时，向待选列表增加全部节点序号
    #         self.c_list.clear()                         # 清空中间变量   ？？？要全部清空么？

    #     return chosen_number


class Select_Node(object):
    def __init__(self, args):
        self.args = args
        self.s_list = []   # 存储待选择的节点名单
        self.c_list = []   # 已选择的节点名单
        self.node_list = list(range(args.node_num))
        self.max_lost = args.max_lost   # 最快节点与最慢节点的通讯回合差

        for j in range(self.max_lost):
            self.s_list.extend(self.node_list)
    

    def random_select(self):
        index = random.randrange(len(self.s_list))      # 随机取下标
        chosen_number = self.s_list.pop(index)          # 找到下标对应的节点序号
        self.c_list.append(chosen_number)               # 中间变量，衡量是否全部节点完成一次通讯
        print(self.c_list)

        if len(set(self.c_list)) == self.args.node_num :
            self.s_list.extend(self.node_list)          # 判断为真，代表所有节点完成一次迭代，此时，向待选列表增加全部节点序号
            [self.c_list.remove(i) for i in range(self.args.node_num)]     # 清空中间变量

        return chosen_number


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = init_model(self.args.global_model).to(self.device)

        self.test_data = test_data
        self.Dict = self.model.state_dict()

#         self.edge_node = [Model.ResNet18().to(self.device) for k in range(args.node_num)]
        self.edge_node = [init_model(self.args.global_model).to(self.device) for k in range(args.node_num)]
        self.init = False
        self.save = []

    def merge(self, Edge_nodes):
        # weights_zero(self.model)
        Node_State_List = [copy.deepcopy(Edge_nodes[i].model.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict = Node_State_List[0]

        for key in self.Dict.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict[key] += Node_State_List[i][key]

            self.Dict[key] = self.Dict[key].float()     # 不知道为什么数据类型会发生变化
            self.Dict[key] /= len(Edge_nodes)
        self.model.load_state_dict(self.Dict)


    def update(self, Edge_node):
        ## 中央服务器的局部模型更新
        self.edge_node[Edge_node.num-1] = Edge_node.model

    def init_processing(self):
        assert self.init
        ## warmup
        Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in self.save]
        self.Dict = Node_State_List[0]
        for key in self.Dict.keys():
            if 'num_batches_tracked' in key:
                continue

            for i in range(1, len(Node_State_List)):
                self.Dict[key] += Node_State_List[i][key]

            # self.Dict[key] = self.Dict[key].float()     # 不知道为什么数据类型会发生变化
            # print(self.Dict[key], key)
            self.Dict[key] /= float(len(Node_State_List))

        self.model.load_state_dict(self.Dict)

    def processing(self):
        ## 中央服务器的全局模型更新
        Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in range(self.args.node_num)]
        self.Dict = Node_State_List[0]
        for key in self.Dict.keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(1, self.args.node_num):
                self.Dict[key] += Node_State_List[i][key]
            # self.Dict[key] = self.Dict[key].float()     # 不知道为什么数据类型会发生变化
            self.Dict[key] /= self.args.node_num
        self.model.load_state_dict(self.Dict)

        


# class Initiate_node(object):
#     def __init__(self, test_data, args):
#         self.num = 0
#         self.args = args
#         self.device = self.args.device
#         self.model = init_model(self.args.global_model).to(self.device)


#         self.test_data = test_data
#         self.Dict = self.model.state_dict()



#     def merge(self, Edge_nodes):
#         # weights_zero(self.model)
#         Node_State_List = [copy.deepcopy(Edge_nodes[i].global_model.state_dict()) for i in range(len(Edge_nodes))]
#         self.Dict = Node_State_List[0]

#         for key in self.Dict.keys():
#             for i in range(1, len(Edge_nodes)):
#                 self.Dict[key] += Node_State_List[i][key]

#             self.Dict[key] = self.Dict[key].float()     # 不知道为什么数据类型会发生变化
#             self.Dict[key] /= len(Edge_nodes)
#         self.model.load_state_dict(self.Dict)

