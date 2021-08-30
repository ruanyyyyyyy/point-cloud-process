# import pointnet2_lib.tools._init_path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from data.resampled_dataset import KITTIPCDClsDataset_Wrapper
from models.pointnet2_msg_cls import PointNet2ClassificationMSG
import argparse
import importlib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_sched

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_save_interval", type=int, default=5)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--ckpt", type=str, default='None')

parser.add_argument("--net", type=str, default='pointnet2_msg_cls')

parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--lr_decay', type=float, default=0.7)
parser.add_argument('--bn_momentum', type=float, default=0.5)
parser.add_argument('--bnm_decay', type=float, default=0.5)
parser.add_argument('--decay_step', type=float, default=2e4)

parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument("--output_dir", type=str, default='../output')
parser.add_argument("--extra_tag", type=str, default='default')

parser.add_argument("--resume", type=str, default='false')

args = parser.parse_args()

FG_THRESH = 0.3

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


def log_print(info, log_f=None):
    print(info)
    if log_f is not None:
        print(info, file=log_f)

def configure_optimizers(config, global_step, model):
    lr_clip = 1e-5
    bnm_clip = 1e-2
    lr_lbmd = lambda _: max(config.lr_decay** (int(global_step * config.batch_size / config.decay_step)),
        lr_clip / config.lr,
    )
    bn_lbmd = lambda _: max(
        config.bn_momentum
        * config.bnm_decay
        ** (int(global_step  * config.batch_size / config.decay_step)),
        bnm_clip,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
    bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd)

    return [optimizer], [lr_scheduler, bnm_scheduler]

def train_one_epoch(model, train_loader, optimizer, lr_scheduler, epoch, total_it, tb_log, log_f):
    model.train()
    log_print('===============TRAIN EPOCH %d================' % epoch, log_f=log_f)
    loss_func = nn.CrossEntropyLoss()

    for it, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        pts_input, cls_labels = batch
        pts_input = pts_input.cuda(non_blocking=True).float()
        cls_labels = cls_labels.cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)

        loss = F.cross_entropy(pred_cls, cls_labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_it += 1

        with torch.no_grad():
            acc = (torch.argmax(pred_cls, dim=1) == cls_labels).float().mean()

        tb_log.add_scalar('train_loss', loss.item(), total_it)
        tb_log.add_scalar('train_acc', acc, total_it)

        if it % 500 == 0:
            log_print('training epoch %d: it=%d/%d, total_it=%d, loss=%.5f, acc=%.3f, lr=%.5f' %
                    (epoch, it, len(train_loader), total_it, loss.item(), acc, lr_scheduler.get_last_lr()[0]), log_f=log_f)
            lr_scheduler.step()

    return total_it


def eval_one_epoch(model, eval_loader, epoch, tb_log, log_f=None):
    model.train()
    log_print('===============EVAL EPOCH %d================' % epoch, log_f=log_f)
    loss_func = nn.CrossEntropyLoss()

    acc_list = []
    loss_list = []
    for it, batch in enumerate(eval_loader):
        pts_input, cls_labels = batch
        pts_input = pts_input.cuda(non_blocking=True).float()
        cls_labels = cls_labels.cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)

        loss = F.cross_entropy(pred_cls, cls_labels)
        loss_list.append(loss.item())
        
        acc = (torch.argmax(pred_cls, dim=1) == cls_labels).float().mean()
        acc_list.append(acc.item())

        if it % 100 == 0:
            log_print('EVAL: it=%d/%d, acc=%.3f' % (it, len(eval_loader), acc), log_f=log_f)

    acc_list = np.array(acc_list)
    avg_acc = acc_list.mean()
    tb_log.add_scalar('eval_acc', avg_acc, epoch)

    loss_list = np.array(loss_list)
    avg_loss = loss_list.mean()
    tb_log.add_scalar('eval_loss', avg_loss, epoch)

    log_print('\nEpoch %d: Average acc (samples=%d): %.6f' % (epoch, acc_list.__len__(), avg_acc), log_f=log_f)
    return avg_acc

def test_one_epoch(model, test_loader, epoch, tb_log, log_f=None):
    model.train()
    log_print('===============TEST EPOCH %d================' % epoch, log_f=log_f)
    loss_func = nn.CrossEntropyLoss()

    y_truths = []
    y_preds = []

    for it, batch in enumerate(test_loader):
        pts_input, cls_labels = batch

        y_truths.append(cls_labels.numpy())

        pts_input = pts_input.cuda(non_blocking=True).float()
        cls_labels = cls_labels.cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)

        indices = torch.argmax(pred_cls, dim=1)

        y_preds.append(indices.cpu().numpy())

    y_truth = np.hstack(y_truths)
    y_pred = np.hstack(y_preds)
    
    # get confusion matrix
    conf_mat = confusion_matrix(y_truth, y_pred)
    # change to percentage:
    conf_mat = np.dot(
        np.diag(1.0 / conf_mat.sum(axis=1)),
        conf_mat
    )
    conf_mat = 100.0 * conf_mat

    with open(os.path.join("./data/resampled_KITTI/object_names.txt")) as f:
        labels = [l.strip() for l in f.readlines()]

    plt.figure(figsize = (10, 10))
    sn.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title('KITTI 3D Object Classification -- Confusion Matrix')
    plt.savefig('./output/confusion_matrix.png')
    plt.show()
    
    print(
		classification_report(
			y_truth, y_pred, 
			target_names=labels
		)
	)    


def save_checkpoint(model, epoch, ckpt_name):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {'epoch': epoch, 'model_state': model_state}
    ckpt_name = '{}.pth'.format(ckpt_name)
    torch.save(state, ckpt_name)
    log_print(f"save checkpoint to {ckpt_name}")


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def train_and_eval(model, train_loader, eval_loader, tb_log, ckpt_dir, log_f, config):
    model.cuda()
    total_it = 0
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    [optimizer], [lr_scheduler, bnm_scheduler] = configure_optimizers(config, total_it, model)

    for epoch in range(1, args.epochs + 1):
        
        total_it = train_one_epoch(model, train_loader, optimizer, lr_scheduler, epoch, total_it, tb_log, log_f)

        if epoch % args.ckpt_save_interval == 0:
            with torch.no_grad(): 
                avg_acc = eval_one_epoch(model, eval_loader, epoch, tb_log, log_f)
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch)
                save_checkpoint(model, epoch, ckpt_name)


if __name__ == '__main__':
    
    model = PointNet2ClassificationMSG()

    dataset_wrapper = KITTIPCDClsDataset_Wrapper('./data/resampled_KITTI', args)
    train_loader, valid_loader, test_loader = dataset_wrapper.get_dataloader()

    if args.mode == 'train':
        # output dir config
        output_dir = os.path.join(args.output_dir, args.extra_tag)
        os.makedirs(output_dir, exist_ok=True)
        ckpt_dir = os.path.join(output_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        tb_log = SummaryWriter('../runs')

        log_file = os.path.join(output_dir, 'log.txt')
        log_f = open(log_file, 'w')

        for key, val in vars(args).items():
            log_print("{:16} {}".format(key, val), log_f=log_f)

        if args.resume == 'true':
            _ = load_checkpoint(model, args.ckpt)

        # train and eval
        train_and_eval(model, train_loader, valid_loader, tb_log, ckpt_dir, log_f, args)
        log_f.close()
    elif args.mode == 'eval':
        epoch = load_checkpoint(model, args.ckpt)
        model.cuda()
        with torch.no_grad():
            avg_acc = eval_one_epoch(model, valid_loader, epoch, log_f)
    elif args.mode == 'test':
        epoch = load_checkpoint(model, args.ckpt)
        model.cuda()
        with torch.no_grad():
            avg_acc = test_one_epoch(model, valid_loader, epoch, log_f)
        
    else:
        raise NotImplementedError

