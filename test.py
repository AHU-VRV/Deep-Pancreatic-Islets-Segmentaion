import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import time
import argparse
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.io as io
from utils.util import calculate_Accuracy, get_model, class_balanced_cross_entropy_loss
from torch.autograd import Variable
import warnings


# --------------------------------------------------------------------------------

models_list = ['AANet']

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch DRIVE Demo')

parser.add_argument('--epochs', type=int, default=500,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--n_channels', type=int, default=3,
                    help='the channel of input img')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='initial learning rate')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='data/',
                    help='dir of the all img')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=16,
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



def fast_test(model, args, image_root, model_name):
    image_path = os.path.join(image_root, 'test/image')
    gt_path = os.path.join(image_root, 'test/label')
    EPS = 1e-12
    softmax = nn.Softmax(dim=1)
    Background_IOU = []
    pancreas_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []
    PPV = []
    F_SCORE = []
    Average_Precision = []


    for i,image_name in enumerate(os.listdir(image_path)):
        index = image_name.rfind('.')
        name_only = image_name[:index]
        print(name_only)

        start = time.time()
        image = cv2.imread(os.path.join(image_path, image_name))
        image = image / np.max(image)
        label = cv2.imread(os.path.join(gt_path, image_name), cv2.IMREAD_GRAYSCALE) / 255.0
        label[label >= 0.5] = 1
        label[label <= 0.5] = 0

        img_input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img_input = Variable(img_input).float()
        img_input  = img_input.cuda()

        model.eval()

        out = torch.zeros([1,2,2048,2048])
        out = Variable(out).float().cuda()
        for ii in range(16):
            for jj in range(16):
                with torch.no_grad():
                    out[:, :, ii * 128:ii * 128 + 128, jj * 128:jj * 128 + 128] = model(img_input[:, :, ii * 128:ii * 128 + 128, jj * 128:jj * 128 + 128])



        pred_score = softmax(out)
        pred_score = pred_score.cpu().data.numpy().squeeze(0)
        y_pred = pred_score[1,:,:]
        y_pred = y_pred.reshape([-1])

        tmp_out = out.cpu().data.numpy().squeeze(0)
        tmp_out = np.argmax(tmp_out, 0)

        if not os.path.exists(r'test_result/%s_%s_%s' % (model_name, args.batch_size, args.epochs)):
            os.mkdir(r'test_result/%s_%s_%s' % (model_name, args.batch_size, args.epochs))
        cv2.imwrite('test_result/%s_%s_%s/' % (model_name, args.batch_size, args.epochs)+ name_only+'_predict' +'.png', tmp_out*255)

        out_ppi = tmp_out.reshape([-1])
        mask_ppi = label.reshape([-1])

        my_confusion = metrics.confusion_matrix(mask_ppi,out_ppi, ).astype(np.float32)
        meanIU, Acc, Se, Sp, IU, ppv, F_score = calculate_Accuracy(my_confusion)

        average_precision = metrics.average_precision_score(mask_ppi, y_pred)
        Auc = roc_auc_score(mask_ppi, y_pred)
        AUC.append(Auc)
        Average_Precision.append(average_precision)

        Background_IOU.append(IU[0])
        pancreas_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)
        PPV.append(ppv)
        F_SCORE.append(F_score)
        end = time.time()

        print(str(i + 1) + r'/' + str(20) + ': ' + '| average_precision: {:.3f} | Se: {:.3f} | ppv: {:.3f} | Auc: {:.3f} |  F_score: {:f} | pancreas_IOU: {:f}'.format(
            average_precision, Se, ppv, Auc, F_score, IU[1]) + '  |  time:%s' % (end - start))


    print('average_precision: %s  |  Se: %s |  ppv: %s |  Auc: %s |  F_score: %s |  pancreas_IOU: %s ' % (
    str(np.mean(np.stack(Average_Precision))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(PPV))),
    str(np.mean(np.stack(AUC))), str(np.mean(np.stack(F_SCORE))), str(np.mean(np.stack(pancreas_IOU)))))

    # store test information
    with open('test_result/%s_%s_%s_%s_test.txt' % (model_name, args.batch_size, args.epochs, args.my_description), 'a+') as f:
        f.write('average_precision: %s  |  Se: %s |  ppv: %s |  Auc: %s |  F_score: %s |  pancreas_IOU: %s ' % (
        str(np.mean(np.stack(Average_Precision))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(PPV))),
        str(np.mean(np.stack(AUC))), str(np.mean(np.stack(F_SCORE))), str(np.mean(np.stack(pancreas_IOU)))))
        f.write('\n\n')

    return np.mean(np.stack(pancreas_IOU))

def test_one_image(image_root,model,criterion,epoch,model_name):
    image_path = os.path.join(image_root, 'test/image')
    gt_path = os.path.join(image_root, 'test/label')
    EPS = 1e-12
    softmax = nn.Softmax(dim=1)
    Background_IOU = []
    pancreas_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []
    PPV = []
    F_SCORE = []
    test_loss = 0.

    for i, image_name in enumerate(os.listdir(image_path)):
        start = time.time()
        image = cv2.imread(os.path.join(image_path, image_name))
        image = image / np.max(image)
        label = cv2.imread(os.path.join(gt_path, image_name), cv2.IMREAD_GRAYSCALE) / 255.0
        label[label >= 0.5] = 1
        label[label <= 0.5] = 0

        target = torch.from_numpy(label).unsqueeze(0)

        # cross entroy loss
        target = Variable(target).long()
        target = target.cuda()

        img_input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        img_input = Variable(img_input).float()
        img_input = img_input.cuda()

        model.eval()

        out = torch.zeros([1, 2, 2048, 2048])
        out = Variable(out).float().cuda()
        for ii in range(16):
            for jj in range(16):
                with torch.no_grad():
                    out[:, :, ii * 128:ii * 128 + 128, jj * 128:jj * 128 + 128] = model(
                        img_input[:, :, ii * 128:ii * 128 + 128, jj * 128:jj * 128 + 128])

        loss = criterion(out, target)
        test_loss += loss.data

        pred_score = softmax(out)
        pred_score = pred_score.cpu().data.numpy().squeeze(0)
        y_pred = pred_score[1, :, :]
        y_pred = y_pred.reshape([-1])

        tmp_out = out.cpu().data.numpy().squeeze(0)
        tmp_out = np.argmax(tmp_out, 0)

        out_ppi = tmp_out.reshape([-1])
        mask_ppi = label.reshape([-1])

        my_confusion = metrics.confusion_matrix(mask_ppi, out_ppi, ).astype(np.float32)
        meanIU, Acc, Se, Sp, IU, ppv, F_score = calculate_Accuracy(my_confusion)
        Auc = roc_auc_score(mask_ppi, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        pancreas_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)
        PPV.append(ppv)
        F_SCORE.append(F_score)
        end = time.time()

        print(str(i + 1) + r'/' + str(
            20) + ': ' + '| Acc: {:.3f} | Se: {:.3f}, ppv: {:3f} |  Auc: {:.3f} | F_score: {:f} | pancreas_IOU: {:f}'.format(
            Acc, Se, ppv, Auc, F_score, IU[1]) + '  |  time:%s' % (end - start))

    test_loss = test_loss / len(os.listdir(image_path))
    print(str('test_epoch{:d} | test_loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | ppv: {:.3f}'
              '| Auc: {:f} | F_score: {:f} | pancreas_IOU: {:f}').format(epoch, test_loss, np.mean(np.stack(ACC)), np.mean(np.stack(SE)),np.mean(np.stack(PPV)),
                                                                         np.mean(np.stack(AUC)), np.mean(np.stack(F_SCORE)), np.mean(np.stack(pancreas_IOU))))
    print('*' * 50)

    # store test information
    with open(r'logs/%s_%s_%s_%s.txt' % (model_name, args.batch_size, args.epochs, args.my_description), 'a+') as f:
        f.write('epoch%s | Acc: %s  |  Se: %s |  ppv: %s | F_score: %s |  Auc: %s | pancreas_IOU: %s ' % (
            str(epoch), str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(PPV))),
            str(np.mean(np.stack(F_SCORE))), str(np.mean(np.stack(AUC))), str(np.mean(np.stack(pancreas_IOU)))))
        f.write('\n\n')

    return np.mean(np.stack(pancreas_IOU)), np.mean(np.mean(np.stack(SE))),test_loss




if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    model_name = models_list[args.model_id]

    model = get_model(model_name)
    model = model(num_classes=args.n_class, num_channels=args.n_channels)

    if args.use_gpu:
        model.cuda()
    if True:

        model_path = 'weights/AANet.pth'
        #
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        # model.load_state_dict(torch.load(model_path))
        print('success load models: %s_%s' % (model_name, args.my_description))

    print('This model is %s_%s_%s' % (model_name, args.batch_size, args.epochs))

    fast_test(model, args, args.data_path, model_name)
