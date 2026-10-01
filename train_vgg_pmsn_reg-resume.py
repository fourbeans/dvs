import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import myTransform
from functions import TET_loss, seed_all, get_logger
from torch.utils.tensorboard import SummaryWriter
from sj.spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from neurons_reg import VGGPMSN_reg, PMSNSJ
from models import VGGPSN
import torch.nn.functional as F

# print(args)
# 定义递归函数来查找模块
def find_modules(module, module_type, module_name=''):
    modules = []
    if isinstance(module, module_type) and (module_name == '' or module_name == module._get_name()):
        modules.append(module)
    for name, child_module in module.named_children():
        modules.extend(find_modules(child_module, module_type, module_name))
    return modules

def activation_guided_l1_regularization(weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """
    Compute activity-aware L1 regularization for SNNs.
    
    Args:
        weight: Tensor of shape [C_out, ...], i.e., conv or linear weights
        activation: Tensor of shape [B, C_out, ...], binary spike activations for each sample
    
    Returns:
        Scalar regularization loss
    """
    # Flatten weights to [C_out, D] and compute L1 norms
    W = weight.view(weight.size(0), -1)  # [C_out, D]
    W_l1 = torch.norm(W, p=1, dim=1)     # [C_out]

    B = 1          # batch size
    C_out = activation.size(0)

    total_reg = 0.0

    active_idx = (activation > 0)
    inactive_idx = ~active_idx

    max_active = W_l1[active_idx].max()
    align_penalty = F.relu(max_active - W_l1[inactive_idx]).sum()

    reg = W_l1.sum() / W_l1.numel() + align_penalty / W_l1.numel()
    
    total_reg += reg

    return total_reg / B
def train(model, device, train_loader, criterion, optimizer, epoch, args):
    num_classes = 10
    top_k = 2
    
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    mlif_modules = find_modules(model, PMSNSJ, 'PMSNSJ')
    for module in mlif_modules:
        module.neuron_m.data = (module.neuron_m + module.memory_loss * optimizer.param_groups[0]['lr']) / (1+optimizer.param_groups[0]['lr'])
        module.memory_loss.data = module.memory_loss.data * 0
    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)
        
        outputs,features = model(images)
        mean_out = outputs.mean(1) # 时间步维度求均值
        
        # print(mean_out.shape, labels.shape)
        rights = mean_out.argmax(1) == labels
        
        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)
            
        # 聚类中心的索引
        feature_count = 0
        reg_loss = 0
        for name, layer in model.named_modules():
            if feature_count >= len(features):
                break
            if isinstance(layer, nn.Conv2d) and 'fwd.module.0' in name:  # 卷积层
                # print(f'Convolutional layer: {name}, weight shape: {layer.weight.shape}')
                # print(f'Convolutional layer: {name}, weight shape: {layer.weight.grad}')
                
                if layer.weight.grad is not None:
                    # print(f"Layer: {name}, Gradient shape: {layer.weight.grad.shape}")
                    cluster_centers = features[feature_count]
                    param = layer.weight # 256,3,3,3
                    reg_grad = torch.zeros_like(param.grad)
                    sample_rights = 0
                    for j in range(len(rights)):
                        if rights[j]:
                            # print(rights.shape, cluster_centers.shape, rights, cluster_centers, param.shape)
                            # top k 不是百分比
                            k = int(cluster_centers[j].numel() * (1 / num_classes * top_k))  
                            threshold = torch.topk(cluster_centers[j].flatten(), k, largest=True)[0][-1]  # 获取第 k 大的值作为阈值
                            cluster_centers[j] = (cluster_centers[j] >= threshold).float().detach()
                            
                            reg_loss += activation_guided_l1_regularization(param.data, cluster_centers[j])
                            cluster_centers[j] = torch.zeros(1)
                    feature_count += 1
                # print(name)
            elif isinstance(layer, nn.Linear) and 'classifier' in name:  # 线性层, 这部分每次换网络都得重新写
                if layer.weight.grad is not None:
                    cluster_centers = features[feature_count]
                    param = layer.weight
                    reg_grad = torch.zeros_like(param.grad)
                    sample_rights = 0
                    for j in range(len(rights)):
                        if rights[j]:
                            cluster_centers[j][cluster_centers[j]>0] = 1
                            reg_loss += activation_guided_l1_regularization(param.data, cluster_centers[j])
                            
                            cluster_centers[j] = torch.zeros(1) 
                    feature_count += 1
                # print(name)
        
        loss = loss + 1e-6 * reg_loss / (feature_count+0.0001)
            
        running_loss += loss.item()
        
        
        
        
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
        

    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        
        # t, b, c, h, w = inputs.shape
        # inputs = F.interpolate(inputs.view(-1, c, h, w), size=(48, 48), mode='bilinear', align_corners=False)
        # inputs = inputs.view(t, b, c, 48, 48)
        
        outputs,features = model(inputs)
        mean_out = outputs.mean(1)  # 时间步维度
        
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc

def build_dvscifar():
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(),])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
        # Cutout(n_holes=1, length=16)])
    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48))])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    train_set = CIFAR10DVS(root='/zjh/data/cifar10dvs', train=True, data_type='frame', frames_number=args.T, split_by='number', transform=transform_train)
    test_set = CIFAR10DVS(root='/zjh/data/cifar10dvs', train=False, data_type='frame', frames_number=args.T, split_by='number',  transform=transform_test)

    return train_set, test_set

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j',
                        '--workers',
                        default=0,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--batch_size',
                        default=16,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning_rate',
                        default=0.001,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--seed',
                        default=100,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-T',
                        default=4,
                        type=int,
                        metavar='N',
                        help='snn simulation time (default: 2)')
    parser.add_argument('--means',
                        default=1.0,
                        type=float,
                        metavar='N',
                        help='make all the potential increment around the means (default: 1.0)')
    parser.add_argument('--lamb',
                        default=1e-3,
                        type=float,
                        metavar='N',
                        help='adjust the norm factor to avoid outlier (default: 0.0)')
    parser.add_argument('-out_dir', default='./logs/', type=str, help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-method', type=str, default='VGGPMSN_reg_reg', help='use which network')
    parser.add_argument('-opt', type=str, default='SGD0.1', help='optimizer method')
    parser.add_argument('-tau', type=float, default=0.25, help='tau of LIF')
    parser.add_argument('-TET', action='store_true', help='use the tet loss')
    parser.add_argument('-fixed', action='store_true')
    args = parser.parse_args()
    seed_all(args.seed)
    train_dataset, val_dataset = build_dvscifar()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    print(len(train_loader))
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    print(len(test_loader))


    if args.method == 'PSN':
        model = VGGPSN()
    else:
        model = VGGPMSN_reg(T=args.T)


    # parallel_model = torch.nn.DataParallel(model)
    parallel_model = model
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'SGD0.1':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    elif args.opt == 'SGD0.02':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)
    elif args.opt == 'SGD0.00001':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    log_file_name = f'T{args.T}_opt_{args.opt}_tau_{args.tau}_method_{args.method}_b_{args.batch_size}'
    if args.TET:
        log_file_name += '_TET'

    num_gpus = torch.cuda.device_count()
    log_file_name += f'_{num_gpus}gpu'


    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    out_dir = os.path.join(args.out_dir,
                           log_file_name)

    if args.resume:
        print('load resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # start_epoch = checkpoint['epoch'] + 1
        # best_acc = checkpoint['best_acc']

    logger = get_logger(log_file_name + '.log')
    logger.info('start training!')

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):


        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        scheduler.step()
        facc = test(parallel_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))
        writer.add_scalar('test_acc', facc, epoch)

        save_max = False
        if best_acc < facc:
            best_acc = facc
            save_max = True
            best_epoch = epoch + 1
            # torch.save(parallel_model.module.state_dict(), 'VGGPMSN_reg_woAP.pth')
        logger.info('Best Test acc={:.3f}'.format(best_acc))
        print('\n')

        checkpoint = {
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            # 'epoch': epoch,
            # 'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

