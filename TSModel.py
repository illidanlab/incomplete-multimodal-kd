from model.DNN import DNNNetOne, DNNNetTwo
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from model import Dataset as DS
import os
import pickle

seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# train teacher
class Teacher:
    def __init__(self, input_size, hidden_size, num_classes, device):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

    def train(self, trainloader, epoch_num):
        # set model to training mode
        net = DNNNetOne(self.input_size, self.hidden_size, self.num_classes)
        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(),
                                lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=0, amsgrad=False)
        net.train()
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        return net


# use teacher to label the data
class TeacherLabel:
    def __init__(self, model):
        self.model = model

    def label(self, x):
        self.model.eval()
        predict = self.model(x)
        return predict


# train student
class Student:
    def __init__(self, input_size1, input_size2, hidden_size, num_classes, device):
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

    def train(self, trainloader, teacher1, teacher2, alpha,beta, T, epoch_num):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # set model to training mode
        net = DNNNetTwo(self.input_size1, self.input_size2, self.hidden_size, self.num_classes)
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(),
                                lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=0, amsgrad=False)
        criterion = nn.CrossEntropyLoss()
        net.train()
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                x1, x2, labels = data
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                labels = labels.to(self.device)
                # teachers label
                soft_label1 = TeacherLabel(teacher1).label(x1)
                q1 = F.softmax(soft_label1 / T, dim=1)
                soft_label2 = TeacherLabel(teacher2).label(x2)
                q2 = F.softmax(soft_label2 / T, dim=1)
                q1 = q1.detach()
                q2 = q2.detach()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(x1, x2)
                p = F.log_softmax(outputs / T, dim=1)
                loss1 = criterion(outputs, labels)
                loss2 = nn.KLDivLoss(reduction='batchmean')(p, q1) * (T * T * alpha)
                loss3 = nn.KLDivLoss(reduction='batchmean')(p, q2) * (T * T * beta)
                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
            # print('[%d] loss: %.3f' %
            #     (epoch + 1, running_loss / (i+1)))
        T1_W = teacher1.fc1.weight.data
        T2_W = teacher2.fc1.weight.data
        T1_W_sum = torch.sum(T1_W * T1_W, dim=0)
        T1_W_sum = F.normalize(T1_W_sum, p=2, dim=0)
        T2_W_sum = torch.sum(T2_W * T2_W, dim=0)
        T2_W_sum = F.normalize(T2_W_sum, p=2, dim=0)
        S1_W = net.fc11.weight
        S2_W = net.fc21.weight
        S1_W_sum = torch.sum(S1_W * S1_W, dim=0)
        S1_W_sum = F.normalize(S1_W_sum, p=2, dim=0)
        S2_W_sum = torch.sum(S2_W * S2_W, dim=0)
        S2_W_sum = F.normalize(S2_W_sum, p=2, dim=0)
        return net


class TeacherStudent:
    def __init__(self, nclasses, ninput1, nhidden1, ninput2, nhidden2, nhidden, device, model_dir):
        self.nclasses = nclasses
        self.ninput1 = ninput1
        self.nhidden1 = nhidden1
        self.ninput2 = ninput2
        self.nhidden2 = nhidden2
        self.nhidden = nhidden
        self.device = device
        self.model_dir = model_dir

    def train(self, trainloader1, trainloader2, x1, x2, y, alpha, beta,
              temperature, batch_size, epoch_num):
        if not os.path.isfile(self.model_dir + 'Te1.pkl'):
            Mt1 = Teacher(self.ninput1, self.nhidden1, self.nclasses,
                          self.device)  # teacher model 1
            teacher1 = Mt1.train(trainloader1, epoch_num)
            with open(self.model_dir + 'Te1.pkl', 'wb') as output:
                pickle.dump({'Te1': teacher1}, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.model_dir + 'Te1.pkl', 'rb') as input:
                Te1 = pickle.load(input)
                teacher1 = Te1['Te1']

        if not os.path.isfile(self.model_dir + 'Te2.pkl'):
            Mt2 = Teacher(self.ninput2, self.nhidden2, self.nclasses,
                          self.device)  # teacher model 2
            teacher2 = Mt2.train(trainloader2, epoch_num)
            with open(self.model_dir + 'Te2.pkl', 'wb') as output:
                pickle.dump({'Te2': teacher2}, output, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.model_dir + 'Te2.pkl', 'rb') as input:
                Te2 = pickle.load(input)
                teacher2 = Te2['Te2']

        trainset = DS.DatasetTwo(x1, x2, y)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=1)
        Ms = Student(self.ninput1, self.ninput2, self.nhidden, self.nclasses,
                     self.device)  # student model

        student = Ms.train(trainloader, teacher1, teacher2, alpha, beta, temperature, epoch_num)
        return student, teacher1, teacher2



