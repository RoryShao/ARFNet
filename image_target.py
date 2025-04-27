import argparse
import os
import os.path as osp
import numpy as np
import torch
import random
import datetime
import torch.nn as nn
import torch.optim as optim
from networks import network
from pseudo_label import  obtain_label, obtain_label_multi_centriods
from networks.transformer.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_deit_base_patch16_224
from dataloader import target_data_load
import torch.nn.functional as F
from utils.log_util import Logger
from utils.loss import Entropy
from utils.evaluation import cal_acc
from utils.loss import SupConLoss
from utils.memory import MemoryBank
from utils.loss import CrossEntropyLabelSmooth, kdloss
from networks.transformer.arfnet_vit import ARFNet_ViT
import warnings 
warnings.filterwarnings("ignore")

import sys
sys.path.append('./')

def get_average(att_map):
    att_map = att_map.squeeze()
    average_att_map = att_map.mean(axis=0)
    feature = torch.flatten(average_att_map)
    return feature

def intersection_with_indices(arr1, arr2):
    """
    return common index of two array
    """
    res = []
    indices1 = []
    indices2 = []
    
    for i, val in enumerate(arr1):
        if val in arr2:
            res.append(val)
            indices1.append(i)
            indices2.append(arr2.index(val))
    
    return res, indices1, indices2

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def eval_initial(memory, loader, netF, netB, netC):
    """Initialize the memory bank after one epoch warm up"""
    netF.eval()
    netB.eval()
    # netC.eval()

    features = torch.zeros(memory.num_samples, memory.num_features).cuda()
    labels = torch.zeros(memory.num_samples).long().cuda()
    outputs = torch.zeros(memory.num_samples, args.class_num).cuda()
    with torch.no_grad():
        for i, (imgs, _, idxs) in enumerate(loader):
            imgs = imgs.cuda()
            feature = netB(netF(imgs))
            output = netC(feature)
            features[idxs] = feature
            labels[idxs] = (args.class_num + idxs).long().cuda()
            outputs[idxs] = torch.softmax(output,dim=-1)
            
        for i in range(args.class_num):
            rank_out = outputs[:,i]
            _,r_idx = torch.topk(rank_out,args.K)
            labels[r_idx] = i

        memory.features = F.normalize(features, dim=1)
        memory.labels = labels
    del features,labels,outputs


def train_target(args, log):
    dset_loaders, train_size = target_data_load(args)
    # set base network
    if args.net[0:3].lower() == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'cls':
        netF = network.ClSeResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'arf':
        netF = network.ARFNetBase(res_name=args.net, class_num=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:8].lower() == 'vit_base':
        netF = vit_base_patch16_224(pretrained=True).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.head.out_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:9].lower() == 'avit_base':
        netF = ARFNet_ViT(vit_model='vit_base', pretrained=True, num_classes=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.head.out_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    #create memory bank:
    memory = MemoryBank(args.bottleneck, train_size, args,
                            temp=args.temp, momentum=args.momentum).cuda()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
   
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
 
    max_iter = args.max_epoch * len(dset_loaders["target"]) 
    interval_iter = max_iter // args.interval  
    iter_num = 0

    netF.train()
    netB.train()
    best_acc = 0


    while iter_num < max_iter:        
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test) 

        if inputs_test[0].size(0) == 1:
            continue
        
        if iter_num % interval_iter == 0 and args.cls_par > 0:   
            netF.eval()
            netB.eval()
           
            if args.label_ways == 'dynamic_label':
                mem_label = obtain_label_multi_centriods(dset_loaders["test"], netF, netB, netC, args, log)
            elif args.label_ways == 'shot_label':
                mem_label = obtain_label(dset_loaders["test"], netF, netB, netC, args, log)
            else:
                assert 'Not implement'
            mem_label = torch.from_numpy(mem_label).cuda()

            netF.train()
            netB.train()
 
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            inputs_gloabl, inputs_local = inputs_test[0], inputs_test[1]
            inputs_gloabl = inputs_gloabl.cuda()
            inputs_local = inputs_local.cuda()
            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            inputs_images = torch.cat([inputs_gloabl, inputs_local], dim=0)

            if args.distill:  
                attention_embedding, y1, y2, y3 = netF(inputs_images, distill=args.distill)
            else:
                attention_embedding = netF(inputs_images)

            attention_features = netB(attention_embedding)
            outputs = netC(attention_features)

            if args.cls_par > 0:
                pred = mem_label[tar_idx]
                pred = torch.cat([pred, pred], dim=0)
                classifier_loss = nn.CrossEntropyLoss()(outputs, pred.to(torch.int64))
                classifier_loss *= args.cls_par
                if iter_num < interval_iter and args.dset == "VISDA-C":
                    classifier_loss *= 0
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs)
                entropy_loss = torch.mean(Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)  #
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
                classifier_loss += im_loss
       
            if args.glcl:
                pred = mem_label[tar_idx]                
                bs = inputs_test[0].size(0)
                f1, f2 = torch.split(attention_features, [bs, bs], dim=0)
                f1 = F.normalize(f1)
                f2 = F.normalize(f2)
                features = torch.cat([f2.unsqueeze(1) ,f1.unsqueeze(1)], dim=1)
                criterion = SupConLoss(temperature=args.tau)
                pred = mem_label[tar_idx]
                # cl_loss = memory(F.normalize(f1, dim=1),F.normalize(f2, dim=1), tar_idx, args.k_nbor)
                # classifier_loss += args.lamda_m * cl_loss
                cl_loss = criterion(features)
                classifier_loss += args.glcl_co * cl_loss

            if args.distill:
                pred = mem_label[tar_idx]
                pred = torch.cat([pred, pred], dim=0)
                feas1_loss = kdloss(y1, outputs.detach()) 
                feas2_loss = kdloss(y2, outputs.detach())      
                feas3_loss = kdloss(y3, outputs.detach())   
                feas_loss = feas1_loss + feas2_loss + feas3_loss
                classifier_loss += 0.1 * feas_loss
            
        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(classifier_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            classifier_loss.backward()
            optimizer.step()

        if iter_num==interval_iter:
            eval_initial(memory, dset_loaders["eval"], netF, netB, netC)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                if best_acc < acc_s_te:
                   best_acc = acc_s_te
                log.logger.info('Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list)
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                if best_acc < acc_s_te:
                   best_acc = acc_s_te
                log.logger.info('Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te))

            netF.train()
            netB.train()
    log.logger.info("Best Acc=%.2f" % best_acc + '\n')

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT-TS2C')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31', choices=['VISDA-C', 'office31', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit_base', choices=['vgg16','resnet50', 'resnet101', 'ARFNet50','ARFNet101','SeResnet50', 'SeResnet101', 'ClSeResnet50', 'ClSeResnet101', 'vit_base', 'avit_base'])
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lamb', type=float, default=0.9)
    parser.add_argument('--smooth', type=float, default=0.1)   
 
    parser.add_argument('--ent', action='store_true', default=False)
    parser.add_argument('--gent', action='store_true', default=False)
    parser.add_argument('--glcl', action='store_true', default=False)
    parser.add_argument('--distill', action='store_true', default=False) 

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=0.07)

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)    
    parser.add_argument('--glcl_co', type=float, default=1.0)

    parser.add_argument('--label_ways', type=str, default="shot_label", choices=['dynamic_label', 'shot_label'])
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='checkpoint')
    parser.add_argument('--output_src', type=str, default='checkpoint')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--issave', type=bool, default=True)

    args = parser.parse_args()

    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = args.data_root + args.dset # + '/'
    if os.path.islink(folder):
        folder = os.readlink(folder)
    else:
        folder = folder + '/'
    print(folder)

    log_path = './log/' + args.dset + '/target/'
    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path+'/%s_%s_gpu_%s_log.txt' % (args.dset, date, args.gpu_id), level='info')
    log.logger.info(args)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.s_dset_path = folder + names[args.s] + '.txt'
        args.t_dset_path = folder + names[args.t] + '.txt'
        args.test_dset_path = folder + names[args.t] + '.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.net, names[args.s][0].upper())
        print(args.output_dir_src)
        args.output_dir = osp.join(args.output, args.da, args.dset, args.net, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)

        log.logger.info('+++++++++++++++++++++++++++ Task: {}'.format(args.name))
        train_target(args, log)
