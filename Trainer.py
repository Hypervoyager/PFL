import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Node import Node
# from losses import GCELoss


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def train_normal(node):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_avg(node):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_mutual(node):
    node.model.to(node.device).train()
    node.global_model.to(node.device).train()
    train_loader = node.train_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0
    acc_local = 0.0
    total_global_loss = 0.0
    avg_global_loss = 0.0
    correct_global = 0.0
    acc_global = 0.0
    description = 'Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_global={:.4f} acc_global={:.2f}%'
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            node.global_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_local_loss, acc_local, avg_global_loss, acc_global))
            data, target = data.to(node.device), target.to(node.device)
            output_local = node.model(data)
            output_global = node.global_model(data)
            ce_local = CE_Loss(output_local, target)
            kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_global.detach()))
            ce_global = CE_Loss(output_global, target)
            kl_global = KL_Loss(LogSoftmax(output_global), Softmax(output_local.detach()))
            loss_local = node.args.alpha * ce_local + (1 - node.args.alpha) * kl_local
            loss_global = node.args.beta * ce_global + (1 - node.args.beta) * kl_global
            loss_local.backward()
            loss_global.backward()
            node.optimizer.step()
            node.global_optimizer.step()

            ## loss与acc计算
            total_local_loss += loss_local
            avg_local_loss = total_local_loss / (idx + 1)
            pred_local = output_local.argmax(dim=1)
            correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            acc_local = correct_local / len(train_loader.dataset) * 100
            total_global_loss += loss_global
            avg_global_loss = total_global_loss / (idx + 1)
            pred_global = output_global.argmax(dim=1)
            correct_global += pred_global.eq(target.view_as(pred_global)).sum()
            acc_global = correct_global / len(train_loader.dataset) * 100


def train_coteaching(node, epoch, rate_schedule,R, args):
    node.model.to(node.device).train()
    node.global_model.to(node.device).train()
    train_loader = node.train_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0 
    acc_local = 0.0
    total_global_loss = 0.0
    avg_global_loss = 0.0
    correct_global = 0.0
    acc_global = 0.0
    description = 'Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_global={:.4f} acc_global={:.2f}%'

    # with tqdm(train_loader) as epochs:
    for idx, (data, target) in enumerate(train_loader):

        # epochs.set_description(description.format(node.num, avg_local_loss, acc_local, avg_global_loss, acc_global))
        data, target = data.to(node.device), target.to(node.device)
        output_local = node.model(data)
        output_global = node.global_model(data)

        loss_local, loss_global, overlap = loss_coteaching(output_local, output_global, target, rate_schedule[epoch], args)
        node.optimizer.zero_grad()
        node.global_optimizer.zero_grad()

        loss_local.backward()
        loss_global.backward()
        node.optimizer.step()
        node.global_optimizer.step()
#         print(idx)
#         if epoch ==4:
#             node.overlap_sum += overlap
#             if idx ==78:
#                 node.overlap_rate = node.overlap_sum/10000/rate_schedule[epoch]
#                 print(node.overlap_rate)
#                 node.overlap_sum = 0

            ## loss与acc计算
            # total_local_loss += loss_local
            # avg_local_loss = total_local_loss / (idx + 1)
            # pred_local = output_local.argmax(dim=1)
            # correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            # acc_local = correct_local / len(train_loader.dataset) * 100
            # total_global_loss += loss_global
            # avg_global_loss = total_global_loss / (idx + 1)
            # pred_global = output_global.argmax(dim=1)
            # correct_global += pred_global.eq(target.view_as(pred_global)).sum()
            # acc_global = correct_global / len(train_loader.dataset) * 100


class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'fed_mutual':
            self.train = train_mutual
        elif args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'fed_coteaching':
            self.train = train_coteaching
        elif args.algorithm == 'normal':
            self.train = train_normal

    def __call__(self, node):
        self.train(node)



def loss_coteaching(y_1, y_2, t, forget_rate, args):
    if args.loss == 'CE':
        loss_1 = F.cross_entropy(y_1, t, reduce=False)
        loss_2 = F.cross_entropy(y_2, t, reduce=False)
        
    elif args.loss == 'GCE':
        loss_1 = GCELoss(y_1, t)
        loss_2 = GCELoss(y_2, t)
        
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = torch.argsort(loss_1.data).cuda()

    
    ind_2_sorted = torch.argsort(loss_2.data).cuda()

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * ind_1_sorted.size()[0])

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]



    a = ind_1_update.tolist()
    b = ind_2_update.tolist()
    ovellap = len(set(a) & set(b))


    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    return torch.sum(loss_1_update), torch.sum(loss_2_update), ovellap





