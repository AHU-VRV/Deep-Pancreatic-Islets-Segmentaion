from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter

from networks.aanet import AANet
from utils.loss import dice_bce_loss,dice_loss
from utils.data import ImageFolder
import argparse
import sklearn.metrics as metrics
from utils.util import calculate_Accuracy, get_model, class_balanced_cross_entropy_loss
from test import fast_test,test_one_image




models_list = ['AANet']

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch Pancreas Demo')

parser.add_argument('--epochs', type=int, default=500,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--n_channels', type=int, default=3,
                    help='the channel of input img')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--SEED', type=float, default=10,
                    help='initial seed')

# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='data/pancreas/',
                    help='dir of the all img')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size',type=int, default=16,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=128,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='train',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0',
                    help='the gpu used')

args = parser.parse_args()
# print(args)
# --------------------------------------------------------------------------------


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

EPS = 1e-12

def seed_torch(seed=1):

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# seed_torch(args.SEED)

def CE_Net_Train():
    SE_best = 0.
    pancreas_IOU_best = 0
    model_name = models_list[args.model_id]
    model = get_model(model_name)
    model = model(num_classes=args.n_class, num_channels=args.n_channels)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if args.use_gpu:
        model.cuda()
        print('GPUs used: (%s)' % args.gpu_avaiable)
        print('------- success use GPU --------')



    dataset = ImageFolder(root_path=args.data_path, datasets='pancreas')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()


    if not os.path.exists(r'tensorboard/%s_%s_%s_%s' % (model_name, args.batch_size, args.epochs,args.my_description)):
        os.mkdir(r'tensorboard/%s_%s_%s_%s' % (model_name, args.batch_size, args.epochs, args.my_description))

    writer = SummaryWriter(logdir='tensorboard/%s_%s_%s_%s/metric' % (model_name, args.batch_size, args.epochs, args.my_description))

    print('This model is %s_%s_%s_%s' % (model_name, args.batch_size, args.epochs, args.my_description))
    if not os.path.exists('weights/%s_%s_%s_%s' % (model_name, args.batch_size, args.epochs,args.my_description)):
        os.mkdir('weights/%s_%s_%s_%s' % (model_name, args.batch_size, args.epochs,args.my_description))

    with open(r'logs/%s_%s_%s_%s.txt' % (model_name, args.batch_size, args.epochs,args.my_description), 'w+') as f:
        f.write('This model is %s_%s_%s_%s: ' % (model_name, args.batch_size, args.epochs, args.my_description) + '\n')
        f.write('args: ' + str(args) + '\n')
        f.write('train lens: ' + str(len(dataset)) + ' | test lens: ' + str(len(dataset)))
        f.write('\n\n---------------------------------------------\n\n')

    # weight_dir = 'weights_new/Res_UNet_Backbone2_PAM_Channel_16_500_train/130_best.pth'
    # if os.path.exists(weight_dir):
    #     checkpoint = torch.load(weight_dir)
    #     model.load_state_dict((checkpoint['model']))
    #     optimizer.load_state_dict((checkpoint['optimizer']))
    #     start_epoch = checkpoint['epoch'] + 1
    # else:
    #     start_epoch = 1
    #     print('no saved model,retrain')

    for epoch in range(1, args.epochs + 1):


        model.train()
        train_epoch_loss = 0
        IOU_train = 0
        begin_time = time.time()


        for i, (img, mask) in enumerate(data_loader):

            img, mask = img.cuda(), mask.cuda()

            #without side out
            optimizer.zero_grad()
            out = model.forward(img)

            #cross_entory loss
            mask = mask.squeeze(1)
            loss = criterion(out,mask.long())

            train_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            out_cpu = out.cpu().data.numpy()

            tmp_out = np.argmax(out.cpu().data.numpy(), 1)

            tmp_mask = mask.cpu().data.numpy()

            out_ppi = tmp_out.reshape([-1])
            mask_ppi = tmp_mask.reshape([-1])

            my_confusion = metrics.confusion_matrix(mask_ppi, out_ppi).astype(np.float32)
            meanIU, Acc, Se, Sp, IU, ppv, F_score = calculate_Accuracy(my_confusion)


            print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | PPV: {:.3f} | F_score: {:.3f}'
                      '|pancreas_IOU: {:f}').format(model_name, args.my_description, epoch, i,
                                                                         loss.data, Acc, Se, ppv,F_score,
                                                                         IU[1]))
            IOU_train += IU[1]
        train_epoch_loss = train_epoch_loss / len(data_loader)
        IOU_train = IOU_train / len(data_loader)
        print(str('train_epoch_loss: {:f}, training finish, time: {:1f} s').format(train_epoch_loss, (time.time() - begin_time)))


        writer.add_scalar(' train_Loss', train_epoch_loss, epoch)
        writer.add_scalar(' pancreas_IOU', IOU_train, epoch)


        IOU_test, SE, test_loss = test_one_image(image_root='data/pancreas/', model=model,epoch=epoch,model_name=model_name,scales=None)
        if epoch > 20 and IOU_test > pancreas_IOU_best:
            pancreas_IOU_best = IOU_test
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, 'weights/%s_%s_%s_%s/%s_best.pth' % (model_name, args.batch_size, args.epochs, args.my_description, str(epoch)))
            print('success save Best model')

        print('*' * 50)

        writer.add_scalar(' pancreas_IOU', IOU_test,epoch)
        writer.add_scalar(' test_loss', test_loss, epoch)

    writer.close()

if __name__ == '__main__':
    CE_Net_Train()



