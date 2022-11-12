import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from model import *
from tqdm import tqdm
import os
import csv
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
DataSet_path = "./DataSet.csv"
txt_path = "./data_list.txt"
Batch_size = 32
Learning_rate=0.001
Epoch = 50

def Train(dataloader, model, criterion, optimizer):
        loss, current, n = 0.0, 0.0, 0
        for step, (input,target) in tqdm(enumerate(dataloader)):
            #print(input)
            #print(target)
            input = input.float().to(device)
            #target = torch.tensor(target).to(device)
            target = target.to(device)
            output = model(input).cuda()
            loss = criterion(output, target)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(target==pred)/output.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss.item()
            current += cur_acc.item()
            n = n+1
        train_loss = loss / n
        train_acc = current/ n
        print('train_loss = ' + str(train_loss))
        print('train_acc = ' + str(train_acc))
        return train_loss, train_acc   

def Val(dataloader, model, criterion):
        model.eval()
        loss, current, n = 0.0, 0.0, 0
        with torch.no_grad():
            for step, (input,target) in tqdm(enumerate(dataloader)):
                input, target = input.float().to(device), target.to(device)
                output = model(input).cuda()
                cur_loss = criterion(output, target)
                _, pred = torch.max(output, axis=1)
                cur_acc = torch.sum(target == pred) / output.shape[0]
                loss += cur_loss.item()
                current += cur_acc.item()
                n = n + 1
        val_loss = loss / n
        val_acc = current / n
        print('val_loss' + str(val_loss))
        print('val_acc' + str(val_acc))
        return val_loss, val_acc   

def Test(dataloader, model, criterion):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for step, (input,target) in tqdm(enumerate(dataloader)):
            input, target = input.float().to(device), target.to(device)
            # Test data
            output = model(input).cuda()
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        print("Test_average_loss : {:.4f} , Accuracy : {:.3f}%\n".format(test_loss, 100 * correct / len(dataloader.dataset)))
        acc = 100 * correct / len(dataloader.dataset)
        
        return test_loss, acc    

        

def PaddingOrTruncate(sample, length=16000):
    if len(sample) < length:
        pad = np.zeros(length - len(sample))
        sample = np.hstack([sample, pad])
    elif len(sample) > length:
        sample = sample[-length:]

    assert len(sample) == length
    return sample


class ReadData(Dataset):
    def __init__(self, dataSet_path, Shuffle = True,mode="train"):
        file = open(dataSet_path,"r")
        reader = csv.reader(file)
        file_name = []
        for tmp in reader:
            if reader.line_num == 1:
                continue
            file_name.append((tmp[0],int(tmp[1])))
        #print(file_name)
    
        if Shuffle:
            random.shuffle(file_name)
        
        if mode == "train":
            self.commands = file_name[:int(0.6*len(file_name))]
        elif mode == "val":
            self.commands = file_name[int(0.6*(len(file_name))):int(0.8*(len(file_name)))]
        elif mode == "test":
            self.commands = file_name[int(0.8*(len(file_name))):]

    def __getitem__(self, index):
        fn, label = self.commands[index]
        wav, sr = librosa.load(fn, sr=16000)
        wav = PaddingOrTruncate(wav)
        feature = librosa.feature.mfcc(wav, sr, n_mfcc=50)
        feature = np.expand_dims(feature, axis=0)
        return feature, label

    def __len__(self):
        return len(self.commands)



def matplot_loss(train_loss, val_loss,Min_Train_loss,Min_Val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('The Loss of Train and Val \n The Min Train_loss is: ' + str(format(Min_Train_loss,'0.8f'))+ '\n The Min Val_loss is: '+ str(format(Min_Val_loss,'.8f')))
    plt.savefig('./Output/Train&Val_Loss.jpg')


def matplot_acc(train_acc, val_acc,Max_Train_acc,Max_Val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('The Acc of Train and Val \n The Max Train_acc is: ' + str(format((Max_Train_acc*100),'0.2f'))+'%' + '\n The Max Val_acc is: '+ str(format((Max_Val_acc*100),'.2f'))+'%')
    plt.savefig('./Output/Train&Val_Acc.jpg')






def main():
    best_acc = 0

    train_set = ReadData(dataSet_path = DataSet_path,mode="train")
    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True, num_workers=1 )

    val_set = ReadData(dataSet_path = DataSet_path, mode="val")
    val_loader = DataLoader(val_set, batch_size=Batch_size, shuffle=True, num_workers=1)

    test_set = ReadData(dataSet_path = DataSet_path, mode="test")
    test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=True, num_workers=1)


    model = ResNet18().to(device)

    #optimizer = torch.optim.Adam(model.parameters(),lr=Learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max = int(Epoch)) 


    criterion = nn.CrossEntropyLoss()


    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    Max_Train_acc = 0
    Max_Val_acc=0
    Min_Train_loss=999
    Min_Val_loss=999


    for epoch in tqdm(range(Epoch)):

        train_loss, train_acc = Train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = Val(val_loader, model , criterion)  

        #lr_scheduler.step()


        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)
        if val_acc > Max_Val_acc:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            Max_Val_acc = val_acc
            print(f"save best model, {epoch+1}epoch")
            torch.save(model, 'save_model/best_model.pth')
        if train_loss < Min_Train_loss:
            Min_Train_loss = train_loss

        if val_loss < Min_Val_loss:
            Min_Val_loss = val_loss
        
        if train_acc > Max_Train_acc:
            Max_Train_acc = train_acc

    #test_loss, test_acc = infer(model, test_loader, criterion)
    #model = ResNet18().to(device)
    model = torch.load("./save_model/best_model.pth")
    test_loss, acc = Test(test_loader,model, criterion)
    matplot_loss(loss_train, loss_val,Min_Train_loss,Min_Val_loss)
    plt.cla()
    matplot_acc(acc_train, acc_val,Max_Train_acc,Max_Val_acc)
    plt.cla()


if __name__ == "__main__":
    main()