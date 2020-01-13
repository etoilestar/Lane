import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from data import MYdata
from config import params
# from network.net import net
from evaluate import Total_loss, cal_iou
from tqdm import tqdm
import segmentation_models_pytorch as smp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def valid(epoch, model, val_data, optimizer, Loss):
    losses = AverageMeter()
    IOU = AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (input, label) in enumerate(tqdm(val_data)):
            input = input.cuda(params['gpu'][0])
            label = label.cuda(params['gpu'][0])
            out = model(input)

            loss, celoss, diceloss = Loss(out, label)

            iou = 0
            for i, (out_batch, label_batch) in enumerate(zip(out, label)):
                out_batch = torch.argmax(out_batch, 0)
                batch_iou = cal_iou(out_batch, label_batch)
                iou += batch_iou
            iou /= i

            losses.update(loss.item())
            IOU.update(iou)
    print('epoch:', epoch, 'valid loss:%0.4f'%losses.avg, celoss.item(), diceloss.item(), 'iou:%0.4f'%IOU.avg, 'lr:', optimizer.param_groups[0]['lr'])
    return losses.avg

def train(epoch, model, train_data, optimizer, Loss):
    losses = AverageMeter()
    IOU = AverageMeter()

    model.train()
    for step, (input, label) in enumerate(tqdm(train_data)):
        input = input.cuda(params['gpu'][0])
        label = label.cuda(params['gpu'][0])
        out = model(input)

        loss, celoss, diceloss = Loss(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = 0
        for i, (out_batch, label_batch) in enumerate(zip(out, label)):
            out_batch = torch.argmax(out_batch, 0)
            batch_iou = cal_iou(out_batch, label_batch)
            iou += batch_iou
        iou /= i

        losses.update(loss.item())
        IOU.update(iou)
    print('epoch:', epoch, 'train loss:%0.4f'%losses.avg, celoss.item(), diceloss.item(), 'iou:%0.4f'%IOU.avg, 'lr:', optimizer.param_groups[0]['lr'])
    return losses.avg

def main():
    train_data = DataLoader(MYdata(params['csv'], mode='train'), batch_size=params['batchsize'], shuffle=True,num_workers=params['num_works'])
    valid_data = DataLoader(MYdata(params['csv'], mode='valid'),batch_size=params['batchsize'],shuffle=False, num_workers=params['num_works'])

    # model = net()
    model = smp.Unet('resnet18', classes=8, encoder_weights='imagenet', activation='softmax')#, activation='softmax'
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])

    if params['pretrain']:
        pretrain_dict = torch.load(params['pretrain'], map_location='cpu')
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=3, verbose=True)
    Loss = Total_loss()

    for n in range(params['max_epoch']):
        train_loss = train(n+1, model, train_data, optimizer, Loss)
        valid_loss = valid(n+1,model, valid_data,optimizer,Loss)

        schedule.step(valid_loss)
        torch.save(model.state_dict(), params['save_path'])

if __name__ == '__main__':
    main()
