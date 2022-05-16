import torch.nn as nn
import torch
from model.sparsemax import Sparsemax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rate = 0.5
class DNNNetOne(nn.Module):
    def __init__(self, input_size, hidden_size, noutput):
        super(DNNNetOne, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, noutput)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        return out

class DNNNetOne_T(nn.Module):
    def __init__(self, input_size, hidden_size, noutput):
        super(DNNNetOne_T, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, noutput)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu(out))
        hidden = out
        out = self.fc3(out)
        return out, hidden


class DNNNetOnelayer(nn.Module):
    def __init__(self, input_size, hidden_size, noutput):
        super(DNNNetOnelayer, self).__init__()
        self.fc1 = nn.Linear(input_size, noutput)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        return out

class DNNNetTwo(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_classes):
        super(DNNNetTwo, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(input_size2, hidden_size)
        self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2):
        out1 = self.fc11(x1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        out2 = self.fc21(x2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out =self.fc3(torch.cat((out1, out2), dim=1))
        return out


class DNNNetTwoDif(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size1, hidden_size2, num_classes):
        super(DNNNetTwoDif, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size1)
        self.fc12 = nn.Linear(hidden_size1, hidden_size1)
        self.fc21 = nn.Linear(input_size2, hidden_size2)
        self.fc22 = nn.Linear(hidden_size2, hidden_size2)
        self.fc3 = nn.Linear(hidden_size1 + hidden_size2, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2):
        out1 = self.fc11(x1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        out2 = self.fc21(x2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out =self.fc3(torch.cat((out1, out2), dim=1))
        return out



class DNNNetOneDecomp(nn.Module):
    def __init__(self, input_size1, hidden_size, num_classes):
        super(DNNNetOneDecomp, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.half = int(self.hidden_size/2)
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1):
        out = self.fc11(x1)
        out = self.dropout(self.relu(out))
        out = self.fc12(out)
        out11 = out[:, :self.half]
        out12 = out[:, self.half:]
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        out = self.dropout(self.relu(out))
        out = self.fc4(out)
        return out, out11, out12


class DNNNetOneAttention(nn.Module):
    def __init__(self, input_size, hidden_size, noutput):
        super(DNNNetOneAttention, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, noutput)
        self.fca1 = nn.Linear(input_size, input_size)
        #self.fca2 = nn.Linear(hidden_size, hidden_size)
        self.sparsemax = Sparsemax(device, dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x):
        # attention1 is a sparse attention
        attention1 = self.sparsemax(self.fca1(x))
        out = self.fc1(x * attention1)
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        out = self.dropout(self.relu(out))
        out = self.fc3(out)
        return out, attention1


class DNNNetTwoAttention(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_classes):
        super(DNNNetTwoAttention, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(input_size2, hidden_size)
        self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, num_classes)

        self.fca1 = nn.Linear(input_size1, input_size1)
        self.fca2 = nn.Linear(input_size2, input_size2)
        self.sparsemax = Sparsemax(device, dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2):
        attention1 = self.sparsemax(self.fca1(x1))
        out1 = self.fc11(x1 * attention1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        attention2 = self.sparsemax(self.fca2(x2))
        out2 = self.fc21(x2 * attention2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out = self.fc3(torch.cat((out1, out2), dim=1))
        return out, attention1, attention2


class DNNNetThree(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, hidden_size, num_classes):
        super(DNNNetThree, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(input_size2, hidden_size)
        self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(input_size3, hidden_size)
        self.fc32 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(3 * hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2, x3):
        out1 = self.fc11(x1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        out2 = self.fc21(x2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out3 = self.fc31(x3)
        out3 = self.dropout(self.relu(out3))
        out3 = self.fc32(out3)
        out3 = self.dropout(self.relu(out3))

        out =self.fc3(torch.cat((out1, out2, out3), dim=1))
        return out


class DNNNetThreeDif(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, hidden_size1,
                 hidden_size2, hidden_size3, num_classes):
        super(DNNNetThreeDif, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size1)
        self.fc12 = nn.Linear(hidden_size1, hidden_size1)
        self.fc21 = nn.Linear(input_size2, hidden_size2)
        self.fc22 = nn.Linear(hidden_size2, hidden_size2)
        self.fc31 = nn.Linear(input_size3, hidden_size3)
        self.fc32 = nn.Linear(hidden_size3, hidden_size3)
        self.fc3 = nn.Linear(hidden_size1 + hidden_size2 + hidden_size3, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2, x3):
        out1 = self.fc11(x1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        out2 = self.fc21(x2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out3 = self.fc31(x3)
        out3 = self.dropout(self.relu(out3))
        out3 = self.fc32(out3)
        out3 = self.dropout(self.relu(out3))

        out =self.fc3(torch.cat((out1, out2, out3), dim=1))
        return out


class DNNNetFive(nn.Module):
    def __init__(self, input_size1, hidden_size, num_classes):
        super(DNNNetFive, self).__init__()
        self.fc11 = nn.Linear(input_size1, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc21 = nn.Linear(input_size1, hidden_size)
        self.fc22 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(input_size1, int(hidden_size/2))
        self.fc32 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc41 = nn.Linear(input_size1, int(hidden_size/2))
        self.fc42 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc51 = nn.Linear(input_size1, int(hidden_size/2))
        self.fc52 = nn.Linear(int(hidden_size/2), int(hidden_size/2))

        self.fc6 = nn.Linear(3 * hidden_size + int(hidden_size/2), num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x1, x2, x3, x4, x5):
        out1 = self.fc11(x1)
        out1 = self.dropout(self.relu(out1))
        out1 = self.fc12(out1)
        out1 = self.dropout(self.relu(out1))

        out2 = self.fc21(x2)
        out2 = self.dropout(self.relu(out2))
        out2 = self.fc22(out2)
        out2 = self.dropout(self.relu(out2))

        out3 = self.fc31(x3)
        out3 = self.dropout(self.relu(out3))
        out3 = self.fc32(out3)
        out3 = self.dropout(self.relu(out3))

        out4 = self.fc41(x4)
        out4 = self.dropout(self.relu(out4))
        out4 = self.fc42(out4)
        out4 = self.dropout(self.relu(out4))

        out5 = self.fc51(x5)
        out5 = self.dropout(self.relu(out5))
        out5 = self.fc52(out5)
        out5 = self.dropout(self.relu(out5))

        out =self.fc6(torch.cat((out1, out2, out3, out4, out5), dim=1))
        return out



