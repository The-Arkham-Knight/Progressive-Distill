import os
import re
import glob
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn import metrics
from efficientnet_pytorch import EfficientNet

from torch.utils.data import Dataset, DataLoader, random_split
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Resize,
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import warnings
import sys
train_transforms = Compose(
    [   LoadImage(image_only=True),
     EnsureChannelFirst(),
        ScaleIntensity(),
        transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 依概率水平旋转
    ]
)
stu_train_transforms = Compose(
    [   LoadImage(image_only=True),
     EnsureChannelFirst(),
        ScaleIntensity(),
        transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 依概率水平旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),#调整对比度、亮度、饱和度、色调
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),#随机擦除
    ]
)
test_transforms = Compose([LoadImage(image_only=True),EnsureChannelFirst(),
        ScaleIntensity(),transforms.Resize((224,224)),
                          ])


root_dir="/home/temp64/SSD2TB/35_250/"
test_dir = "/home/temp64/SSD2TB/35_250/full/"
train_dir = "/home/temp64/SSD2TB/35_250/full/"
test_folders = os.listdir(test_dir)
train_folders = os.listdir(train_dir)
print(len(train_folders))
print(len(test_folders))

# 读取训练集和测试集的Excel文件
train_excel_file = "/home/temp64/SSD2TB/35_250/train.xlsx"
train_df = pd.read_excel(train_excel_file)
train_label_dict = dict(zip(train_df.iloc[:, 0], train_df.iloc[:, 12]))

test_excel_file = "/home/temp64/SSD2TB/35_250/test.xlsx"
test_df = pd.read_excel(test_excel_file)
test_label_dict = dict(zip(test_df.iloc[:, 0], test_df.iloc[:, 12]))

train_data = []  # 创建一个空列表，用来存储训练集的数据
test_data = []  # 创建一个空列表，用来存储测试集的数据
num_0=0
num_1=0
# 遍历训练集的文件夹名称
for folder in train_folders:
    
    patient_name = os.path.basename(folder)  # 获取文件夹的名称，如1zhangbutian
    patient_id = int(re.search(r"\d+", patient_name).group())  # 用正则表达式提取名称中的数字，如1
    if patient_id in train_label_dict:
        
        label = train_label_dict[patient_id]  # 根据字典查找对应的标签
        if label in ["PR", "CR"]:
            num_0+=1
            bag_label = 0
        elif label in ["PD", "SD"]:
            num_1+=1
            bag_label = 1
        else:
            print(label)
            raise ValueError("Invalid label!")
        for subfolder in os.listdir(os.path.join(train_dir, folder)):
            for file in os.listdir(os.path.join(train_dir, folder, subfolder)):
                image = os.path.join(train_dir, folder, subfolder, file)  # 获取图片路径
                train_data.append((patient_id, image, bag_label))  # 将图片路径和标签添加到训练集数据列表中
print(num_0,num_1)
num_0=0
num_1=0
# 遍历测试集的文件夹名称
for folder in test_folders:
    
    patient_name = os.path.basename(folder)  # 获取文件夹的名称，如1zhangbutian
    patient_id = int(re.search(r"\d+", patient_name).group())  # 用正则表达式提取名称中的数字，如1
    if patient_id in test_label_dict:
        label = test_label_dict[patient_id]  # 根据字典查找对应的标签
        if label in ["PR", "CR"]:
            num_0+=1
            bag_label = 0
        elif label in ["PD", "SD"]:
            num_1+=1
            bag_label = 1
        else:
            print(label)
            raise ValueError("Invalid label!")
        for subfolder in os.listdir(os.path.join(test_dir, folder)):
            for file in os.listdir(os.path.join(test_dir, folder, subfolder)):
                image = os.path.join(test_dir, folder, subfolder, file)  # 获取图片路径
                test_data.append((patient_id, image, bag_label))  # 将图片路径和标签添加到测试集数据列表中
print(num_0,num_1)

class WenyiDataset(Dataset):
    def __init__(self, data, transform=None, test=None):
        # root_dir: 图片所在的根目录
        # excel_file: excel表格的路径
        # transform: 图片的预处理或增强方法
        # self.root_dir = root_dir
        self.transform = transform
        self.data = data
        self.test = test
        # self.read_excel(excel_file) # 读取excel表格
    
    def __len__(self):
        # 返回列表的长度，即图片的数量
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.test:
            patient, image, label = self.data[idx]
            if self.transform:
                image = self.transform(image)
            return patient, image, label
        # 根据索引，从列表中取出一对图片路径和标签
        else:
            image, label = self.data[idx]
            # 用PIL.Image.open打开图片，并转换为RGB模式（防止有些图片是灰度图）
            # image = Image.open(image_path)
            # 如果有预处理或增强方法，就对图片进行处理
            if self.transform:
                image = self.transform(image)
            # 返回图片和标签
            return image, label
        
# 使用自定义的数据集类创建数据集对象，传入相应的子数据集和转换操作
train_dataset = WenyiDataset(train_data, transform=train_transforms,test=True)
test_dataset = WenyiDataset(test_data, transform=test_transforms, test=True)
stu_train_dataset=WenyiDataset(train_data, transform=stu_train_transforms,test=True)

# 定义一些参数，如批量大小、是否打乱顺序、是否使用多线程等
BATCH_SIZE = 64
NUM_WORKERS = 4

# 使用torch.utils.data.DataLoader类创建数据加载器对象，传入相应的数据集和参数
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
stu_train_loader = DataLoader(stu_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def _get_stats_by_person_optimalcutoff_topN(n,patient_dict):
    predict = []
    all_label = []
    wrong_predictions = []  # 用于存储预测错误的序号
    right_predictions = []
    ids=[]
    file = open('log.txt', 'a')
    for patient in patient_dict:
        output = np.array(patient_dict[patient]['pred'])
        output.sort()
        predict.append(output[-n:].mean())
        all_label.append(patient_dict[patient]['label'])
        ids.append(patient)
    fpr, tpr, threshold = metrics.roc_curve(all_label, predict)
    y = tpr - fpr
    youden_index = np.argmax(y)
    optimal_cutoff = threshold[youden_index]
    
    predict_binary = np.where(predict > optimal_cutoff, 1, 0)
    acc = (predict_binary == all_label).sum() / len(predict)
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(predict_binary)):
        if predict_binary[i] != all_label[i]:
            wrong_predictions.append(ids[i])
        else:
            right_predictions.append(ids[i])
        if predict_binary[i] ==1 and all_label[i]==1:
            tp+=1
        elif predict_binary[i] ==1 and all_label[i]==0:
            fp+=1
        elif predict_binary[i] ==0 and all_label[i]==0:
            tn+=1
        else:
            fn+=1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    for i in range(len(predict_binary)):
        if predict_binary[i] != all_label[i]:
            wrong_predictions.append(ids[i])
        else:
            right_predictions.append(ids[i])
    

    auc_mean = metrics.auc(fpr, tpr)

    print("Auc Mean:{:.2f} | Person Acc:{:.2f} | Sensitivity:{:.2f} | Specificity:{:.2f} | Cutoff:{:.2f}".format(
        auc_mean, acc, 100*sensitivity, 100*specificity, optimal_cutoff)
    )
    if wrong_predictions:
        print("Wrong Predictions: ", wrong_predictions)
    if right_predictions:
        print("Right Predictions: ", right_predictions)

    file.write(f"Wrong Predictions:{wrong_predictions}\n"
        "Auc Mean:{:.2f} | Person Acc:{:.2f} | Sensitivity:{:.2f} | Specificity:{:.2f} | Cutoff:{:.2f}\n".format(
        auc_mean, acc, 100*sensitivity, 100*specificity, optimal_cutoff))

    return auc_mean, acc

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class student_model(nn.Module):
    def __init__(self, model_name='efficientnet-b0', in_channels=1, num_classes=2, dropout_rate=0.0):
        super(student_model, self).__init__()
        # 创建EfficientNet模型
        self.model = EfficientNet.from_pretrained(model_name, in_channels=in_channels, num_classes=num_classes)
        
        # 添加dropout层，受dropout_rate参数控制
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = self.model(x)
        if self.training:
            x = self.dropout(x)
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        y_hat = torch.softmax(x, dim=1)
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])

from torch.autograd import Variable
class FocalLossWithLabelSmooth(nn.Module):
    def __init__(self,class_num=2, alpha=0.25, gamma=2, smoothing=0.1):
        super(FocalLossWithLabelSmooth, self).__init__()
        self.class_num = class_num
        # for label smooth
        self.smoothing = smoothing
        # for focal loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        label = label.contiguous().view(-1)
        one_hot_label = torch.zeros_like(pred)
        one_hot_label = one_hot_label.scatter(1, label.view(-1, 1), 1)
        one_hot_label = one_hot_label * (1 - self.smoothing) + (1 - one_hot_label) * self.smoothing / (self.class_num - 1)
        # for label smooth
        log_prob = F.log_softmax(pred, dim=1)
        CEloss = (one_hot_label * log_prob).sum(dim=1)
        #print(one_hot_label) 
        # for focal loss
        P = F.softmax(pred, 1)
        class_mask = pred.data.new(pred.size(0), pred.size(1)).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        probs = (P * class_mask).sum(1).view(-1, 1)
        # if multi-class you need to modify here 
        alpha = torch.empty(label.size()).fill_(1 - self.alpha)
        # TODO: multi class
        alpha[label == 1] = self.alpha                                                                                                                     
        
        if pred.is_cuda and not alpha.is_cuda:                                                                                                             
            alpha = alpha.cuda()                                                                                                                           
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * CEloss                                                                                
        loss = batch_loss.mean()                                                                                                                           
        return loss   



device = torch.device("cuda:0")
gpus = [0,1,2]



teacher_model = EfficientNet.from_pretrained('efficientnet-b7', in_channels=1, num_classes=2).to(device)
#teacher_model = student_model('efficientnet-b3', in_channels=1, num_classes=2,dropout_rate=0).to(device)
teacher_model = nn.DataParallel(teacher_model, device_ids= gpus)
student_model1 = student_model('efficientnet-b7', in_channels=1, num_classes=2,dropout_rate=0.2).to(device)
#student_model1 = EfficientNet.from_pretrained('efficientnet-b3', in_channels=1, num_classes=2).to(device)
student_model1 = nn.DataParallel(student_model1, device_ids= gpus)
student_model2 = student_model('efficientnet-b7', in_channels=1, num_classes=2,dropout_rate=0.2).to(device)
#student_model2 = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=2).to(device)
student_model2 = nn.DataParallel(student_model2, device_ids= gpus)
student_model3 = student_model('efficientnet-b0', in_channels=1, num_classes=2,dropout_rate=0).to(device)
student_model3 = nn.DataParallel(student_model3, device_ids= gpus)
student_model4 = student_model('efficientnet-b0', in_channels=1, num_classes=2,dropout_rate=0).to(device)
student_model4 = nn.DataParallel(student_model4, device_ids= gpus)


optimizer_t= torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_s1= torch.optim.SGD(student_model1.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_s2= torch.optim.SGD(student_model2.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_s3= torch.optim.SGD(student_model3.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_s4= torch.optim.SGD(student_model4.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


loss_function = LabelSmoothingCrossEntropy(smoothing=0.3)
#loss_function = FocalLossWithLabelSmooth(alpha=0.75,gamma=2,smoothing=0.3)
#loss_function = torch.nn.CrossEntropyLoss()

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler_t=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=30, eta_min=0.00001)
scheduler_s1=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=30, eta_min=0.00001)
scheduler_s2=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=30, eta_min=0.00001)
scheduler_s3=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=30, eta_min=0.00001)
scheduler_s4=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s4, T_max=30, eta_min=0.00001)
#scheduler_s3=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=30, eta_min=0.00001)

max_epochs = 15
auc_metric = ROCAUCMetric()


#添加Stochastic Depth
drop_prob_s1 = 0.2  # 设置丢弃概率，根据需要进行调整
drop_prob_s2 = 0.2
scale_by_keep = True


# 遍历模型的子模块，找到需要添加Stochastic Depth的位置
for module in student_model1.modules():
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            if i % 2 == 0 and isinstance(sub_module, nn.Module):  # 在每个子模块的第一层添加Stochastic Depth
                sub_module.add_module("drop_path", DropPath(drop_prob_s1, scale_by_keep))
for module in student_model2.modules():
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            if i % 2 == 0 and isinstance(sub_module, nn.Module):  # 在每个子模块的第一层添加Stochastic Depth
                sub_module.add_module("drop_path", DropPath(drop_prob_s2, scale_by_keep))
for module in student_model3.modules():
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            if i % 2 == 0 and isinstance(sub_module, nn.Module):  # 在每个子模块的第一层添加Stochastic Depth
                sub_module.add_module("drop_path", DropPath(0.4, scale_by_keep))
# for module in student_model4.modules():
#     if isinstance(module, nn.Sequential):
#         for i, sub_module in enumerate(module):
#             if i % 2 == 0 and isinstance(sub_module, nn.Module):  # 在每个子模块的第一层添加Stochastic Depth
#                 sub_module.add_module("drop_path", DropPath(drop_prob_s2, scale_by_keep))



warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")
Temp = 2.  # 温度常数
#alpha = 0.7

soft_loss = nn.KLDivLoss(reduction="batchmean")
hard_loss = LabelSmoothingCrossEntropy(smoothing=0.4)  
#hard_loss = FocalLossWithLabelSmooth(alpha=0.75,gamma=2,smoothing=0.4)

def teacher_train(teacher_model, train_loader, test_loader, loss_function, max_epochs,model_name):
    best_metric = -1
    best_acc=-1
    best_metric_epoch = -1
    # train
    
    for epoch in range(max_epochs):
        teacher_model.train()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        file = open('log.txt', 'a')
        file.write(f"epoch {epoch + 1}/{max_epochs}\n")
        for batch_data in train_loader:
            inputs, labels = batch_data[1].to(device), batch_data[2].to(device)
            output = teacher_model(inputs)
            loss = loss_function(output, labels)
            optimizer_t.zero_grad()
            loss.backward()
            optimizer_t.step()
        print("loss: ", loss)
        scheduler_t.step()
        # eval
        correct = 0 # 初始化正确的个数为0
        file = open('log.txt', 'a')
        teacher_model.eval()
        with torch.no_grad():
            patient_dict = {}
            for test_data in test_loader:
                patient_ids, test_images, test_labels = (
                    test_data[0],
                    test_data[1].to(device),
                    test_data[2].to(device)
                )
                pred = teacher_model(test_images)
                softmax_pred = F.softmax(pred, -1)
                _, preds = torch.max(softmax_pred, 1) # 获取预测的类别
                correct += torch.sum(preds == test_labels).item()
                

                for idx in range(softmax_pred.size(0)):
                    if int(patient_ids[idx]) in patient_dict:
                        patient_dict[int(patient_ids[idx])]['pred'].append(softmax_pred[idx][1].detach().cpu())
                        assert patient_dict[int(patient_ids[idx])]['label'] == test_labels[idx] ,'label not match'
                    else:
                        # print(idx)
                        patient_dict[int(patient_ids[idx])] = {'pred':[softmax_pred[idx][1].detach().cpu(),], 'label':int(test_labels[idx])}

            result, acc = _get_stats_by_person_optimalcutoff_topN(10,patient_dict)
            test_acc = correct / len(test_dataset) # 计算训练准确率
            print(f"test accuracy: {test_acc:.4f}") # 输出训练准确率
            file.write(f"test accuracy: {test_acc:.4f}\n")
            if acc > best_acc:
                best_acc = acc
                best_metric=result
                best_metric_epoch = epoch + 1
                torch.save(teacher_model.state_dict(), os.path.join("/home/temp64/SSD2TB/35_250/checkpoint", model_name))
                print("saved new best metric model")
            elif acc==best_acc:
                if result>best_metric:
                    best_metric=result
                    best_metric_epoch=epoch+1
                    torch.save(teacher_model.state_dict(), os.path.join("/home/temp64/SSD2TB/35_250/checkpoint", model_name))
                    print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc:.4f}"
                f" best ACC: {best_acc:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    print(f"train completed, best_metric: {best_metric:.4f} "f"best_acc: {best_acc:.4f} " f"at epoch: {best_metric_epoch}")
    file.write(f"train completed, best_metric: {best_metric:.4f} "f"best_acc: {best_acc:.4f} " f"at epoch: {best_metric_epoch}\n")
    file.close()
    return teacher_model

 
def KD_train(teacher_model, student_model, train_loader, test_loader,is_save,optimizer,scheduler,model_name,alpha):
    teacher_model.eval()
    best_metric = -1
    best_acc=-1
    best_metric_epoch = -1
    epoch_loss_values = []
    correct = 0 # 初始化正确的个数为0
    for epoch in range(max_epochs):
        file = open('log.txt', 'a')
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        file.write(f"epoch {epoch + 1}/{max_epochs}\n")
        student_model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            patient_ids, inputs, labels = batch_data[0], batch_data[1].to(device), batch_data[2].to(device)
            # teacher model
            with torch.no_grad():	# 教师网络不用反向传播
                techer_preds = teacher_model(inputs)

            # student model forward
            student_preds = student_model(inputs)
            student_loss = hard_loss(student_preds, labels)

            ditillation_loss = soft_loss(
                F.log_softmax(student_preds/Temp, dim = 1),
                F.softmax(techer_preds/Temp, dim = 1)
            )

            loss = alpha * student_loss + (1 - alpha) * ditillation_loss * Temp * Temp # 温度的平方
            softmax_pred = F.softmax(student_preds, -1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, preds = torch.max(softmax_pred, 1) # 获取预测的类别
            correct += torch.sum(preds == labels).item() # 累加正确的个数
            if (step) % 30 == 0:
                print(f"{step}/{len(train_dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        # 更新学习率
        scheduler.step()
        
        # 打印每个epoch结束后的学习率
        print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        train_acc = correct / len(train_dataset) # 计算训练准确率
        print(f"epoch {epoch + 1} train accuracy: {train_acc:.4f}") # 输出训练准确率
        
        file.write(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}\n"
            f"epoch {epoch + 1} average loss: {epoch_loss:.4f}\n"
                f"epoch {epoch + 1} train accuracy: {train_acc:.4f}\n")
        correct = 0 # 初始化正确的个数为0

        file = open('log.txt', 'a')
        student_model.eval()
        with torch.no_grad():
            patient_dict = {}
            for test_data in test_loader:
                patient_ids, test_images, test_labels = (
                    test_data[0],
                    test_data[1].to(device),
                    test_data[2].to(device)
                )
                pred = student_model(test_images)
                softmax_pred = F.softmax(pred, -1)
                _, preds = torch.max(softmax_pred, 1) # 获取预测的类别
                correct += torch.sum(preds == test_labels).item()
                

                for idx in range(softmax_pred.size(0)):
                    if int(patient_ids[idx]) in patient_dict:
                        patient_dict[int(patient_ids[idx])]['pred'].append(softmax_pred[idx][1].detach().cpu())
                        assert patient_dict[int(patient_ids[idx])]['label'] == test_labels[idx] ,'label not match'
                    else:
                        # print(idx)
                        patient_dict[int(patient_ids[idx])] = {'pred':[softmax_pred[idx][1].detach().cpu(),], 'label':int(test_labels[idx])}

            result, acc = _get_stats_by_person_optimalcutoff_topN(10,patient_dict)
            test_acc = correct / len(test_dataset) # 计算训练准确率
            print(f"test accuracy: {test_acc:.4f}") # 输出训练准确率
            file.write(f"test accuracy: {test_acc:.4f}\n")
            if is_save==True:
                if acc > best_acc:
                    best_acc = acc
                    best_metric=result
                    best_metric_epoch = epoch + 1
                    torch.save(student_model.state_dict(), os.path.join("/home/temp64/SSD2TB/35_250/checkpoint", model_name))
                    print("saved new best metric model")
                elif acc==best_acc:
                    if result>best_metric:
                        best_metric=result
                        best_metric_epoch=epoch+1
                        torch.save(student_model.state_dict(), os.path.join("/home/temp64/SSD2TB/35_250/checkpoint", model_name))
                        print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc:.4f}"
                f" best ACC: {best_acc:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    print(f"train completed, best_metric: {best_metric:.4f} "f"best_acc: {best_acc:.4f} " f"at epoch: {best_metric_epoch}")
    file.write(f"train completed, best_metric: {best_metric:.4f} "f"best_acc: {best_acc:.4f} " f"at epoch: {best_metric_epoch}\n")
    file.close()
    return student_model

#state_dict=torch.load((os.path.join("/home/temp64/SSD2TB/35_250/checkpoint/best", "Teacher2.pth")))
#teacher_model.load_state_dict(state_dict) 
 
def do_train(teacher_model,train_loader, test_loader, loss_func, epochs):
    #教师训练
    #print("teacher model train")
    Teacher = teacher_train(teacher_model,train_loader, test_loader, loss_func, epochs,"baseline b3 1.pth") 
    print("\n KD model 1  ready train")
    stu1 = KD_train(Teacher, student_model1, train_loader, test_loader,True,optimizer_s1,scheduler_s1,"Stu1.pth",0.5)
    print("\n KD model 2  ready train")
    stu2 = KD_train(stu1, student_model2, train_loader, test_loader,True,optimizer_s2,scheduler_s2,"Stu2.pth",0.3)
    # print("\n KD model 3  ready train")
    # stu3 = KD_train(stu2, student_model3, train_loader, test_loader,True,optimizer_s3,scheduler_s3,"Stu3.pth",0.3)
    # print("\n KD model 4  ready train")
    # stu4 = KD_train(stu3, student_model4, train_loader, test_loader,True,optimizer_s4,scheduler_s4,"Stu4.pth",0.2)

 
if __name__=="__main__":


    do_train(teacher_model, train_loader, test_loader, loss_function, max_epochs)

 
 
 