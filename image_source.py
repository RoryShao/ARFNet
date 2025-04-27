import argparse
import os
from networks.senet import se_resnet
import torch
import random
import datetime
import numpy as np
import os.path as osp
import torch.optim as optim
from networks import network
from utils.loss import CrossEntropyLabelSmooth, kdloss
from networks.transformer.vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_deit_base_distilled_patch16_224
from networks.transformer.arfnet_vit import ARFNet_ViT
from utils.evaluation import cal_acc
from utils.log_util import Logger
from dataloader import soure_data_load
import torch.nn.functional as F
import warnings 
warnings.filterwarnings("ignore")

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


def train_source(args, log):
    dset_loaders = soure_data_load(args)
    ## set base network
    if args.net[0:3].lower() == 'ser':
        netF = se_resnet.se_resnet50(num_classes=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.fc.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3] == 'SeR':
        netF = network.SeResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3] == 'cls':
        netF = network.ClSeResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'arf':
        netF = network.ARFNetBase(res_name=args.net, class_num=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
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
    elif args.net[0:4].lower() == 'deit':
        netF = vit_deit_base_distilled_patch16_224(pretrained=True).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.head.out_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    best_acc = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)  
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)  

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        if args.distill:  
            embedding, y1, y2, y3 = netF(inputs_source, distill=args.distill)
        elif args.net[0:4].lower() == 'deit':
            embedding = netF(inputs_source)[0] 
        else:
            embedding = netF(inputs_source) 
        
        # print(embedding)
        final_features = netB(embedding)
        outputs_source = netC(final_features)
        total_loss = torch.tensor(0.0).cuda()
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source) 
        if args.distill:  
            feas1_loss = kdloss(y1, outputs_source.detach())  
            feas2_loss = kdloss(y2, outputs_source.detach())  
            feas3_loss = kdloss(y3, outputs_source.detach())  
            feas_loss = feas1_loss + feas2_loss + feas3_loss
            total_loss += classifier_loss + 0.1 * feas_loss
        else:
            total_loss += classifier_loss 


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            log.logger.info('Task: {}, Iter:{}/{};  total_loss = {:.2f}%'.format(args.name_src, iter_num, max_iter, total_loss))
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                if best_acc < acc_s_te:
                   best_acc = acc_s_te
                log.logger.info('Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list)
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                if best_acc < acc_s_te:
                   best_acc = acc_s_te
                log.logger.info('Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te))

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
    log.logger.info("Best Acc=%.2f" % best_acc + '\n')

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC


def test_target(args, log):
    dset_loaders = soure_data_load(args)
    if args.net[0:3] == 'ser':
        netF = se_resnet.se_resnet50(num_classes=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.fc.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3] == 'SeR':
        netF = network.SeResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3] == 'ClS':
        netF = network.ClSeResBase(res_name=args.net, cl_co=0.1).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'arf':
        netF = network.ARFNetBase(res_name=args.net, cl_co=args.cl_co, class_num=args.class_num).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    elif args.net[0:3].lower() == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
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
    elif args.net[0:4].lower() == 'deit':
        netF = vit_deit_base_distilled_patch16_224(pretrained=True).cuda()
        netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.head.out_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.dset=='VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
        log.logger.info('\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list)
    else:
        acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
        log.logger.info('\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc))


def main():
    parser = argparse.ArgumentParser(description='ViT-TS2C')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31', choices=['VISDA-C', 'office31', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='ARFNet50', choices=['vgg16','resnet18','resnet34', 'resnet50','resnet101', 'SeResnet50', 
                                                                        'ARFNet34', 'ARFNet50', 'ARFNet101', 'vit_base', 'avit_base', 'deit_base'])
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--cl_co', type=float, default=0.1)  
    parser.add_argument('--distill', action='store_true', default=False) 
    parser.add_argument('--output', type=str, default='checkpoint')
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
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
    # torch.backends.cudnn.deterministic = True
    folder = args.data_root + args.dset  # + '/'
    if os.path.islink(folder):
        folder = os.readlink(folder)
    else:
        folder = folder + '/'
    print(folder)

    log_path = './log/' + args.dset + '/source/'
    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path+'/%s_%s_gpu_%s_log.txt' % (args.dset, date, args.gpu_id), level='info')
    log.logger.info(args)

    args.s_dset_path = folder + names[args.s] + '.txt'
    args.test_dset_path = folder + names[args.t] + '.txt'

    args.output_dir_src = osp.join(args.output, args.da, args.dset, args.net, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    train_source(args, log)
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        args.s_dset_path = folder + names[args.s] + '.txt'
        args.test_dset_path = folder + names[args.t] + '.txt'
        test_target(args, log)


if __name__ == "__main__":
    main()