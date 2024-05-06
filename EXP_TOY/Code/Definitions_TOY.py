import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

###################################################### CONSTANTS ######################################################
T = 100 # temperature, constant for all
DATA_PATH = "../Data/"
MODELS_PATH = "../Models/"
FIGURES_PATH = "../GeneratedFigures/"
RANDOM_SEED = 42

# define all teacher names and all student names for each teacher
TEACHER_TO_STUDENTS = {
    "RT_temp1":["RS_temp1_from_RT_temp1",
                "FS_temp1_from_RT_temp1",
                "RS_tempT_from_RT_temp1",
                "FS_tempT_from_RT_temp1"],
    "FT_temp1":["RS_temp1_from_FT_temp1",
                "FS_temp1_from_FT_temp1",
                "RS_tempT_from_FT_temp1",
                "FS_tempT_from_FT_temp1"],
    "RT_tempT":["RS_temp1_from_RT_tempT",
                "FS_temp1_from_RT_tempT",
                "RS_tempT_from_RT_tempT",
                "FS_tempT_from_RT_tempT"],
    "FT_tempT":["RS_temp1_from_FT_tempT",
                "FS_temp1_from_FT_tempT",
                "RS_tempT_from_FT_tempT",
                "FS_tempT_from_FT_tempT"],
}

################################################### PYTORCH MODULES ###################################################
class FullNetwork(nn.Module):
    """
    Small sigmoid network for TOY dataset.
    """
    
    def __init__(self, num_classes=3):
        """
        Init method.
        Params:
            num_classes: int, 3 for TOY
            dropout: float, 0.5.
        """
        super().__init__()

        # fully connected
        self.classifier = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """Forward method."""
        return self.classifier(x)

class ReducedNetwork(nn.Module):
    """
    Reduced version of FullNetwork to distill knowledge into.
    """
    
    def __init__(self, num_classes=3):
        """
        Init method.
        Params:
            num_classes: int, 3 for TOY
            dropout: float, 0.5.
        """
        super().__init__()

        # fully connected
        self.classifier = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        """Forward method."""
        return self.classifier(x)
    
################################################## LIGHTNING CLASSES ##################################################
class TeacherModel(L.LightningModule):
    """
    Standard lightning model configured to train normally on hard/one-hot labels.
    """

    def __init__(self, architecture_class, training_temperature):
        """
        Init method. 
        Params:
            architecture_class: nn.Module class, here either FullNetwork or ReducedNetwork.
            training_temperature: float, temperature to use when training.
        """
        super().__init__()
        self.network = architecture_class() # initialize the network to use
        self.training_temperature = training_temperature
        self.loss_module = nn.CrossEntropyLoss() # for teacher just use normal cross entropy
        self.cur_iteration_loss = []
        self.epoch_losses = []

    def forward(self, imgs, temperature): # have to explicitly pass temperature
        return self.network(imgs) / temperature

    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [{"scheduler":scheduler, "interval":"epoch"}]
        
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs, temperature=self.training_temperature) # FullNetwork and ReducedNetwork both return logits
        loss = self.loss_module(logits, labels)
        self.cur_iteration_loss.append(loss.item())
        self.log(f"train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.epoch_losses.append(sum(self.cur_iteration_loss)/len(self.cur_iteration_loss))
        self.cur_iteration_loss = []
    
class StudentModel(L.LightningModule):
    """
    Lightning model configured to train on soft labels given a teacher model.
    """

    def __init__(self, architecture_class, training_temperature, teacher_model):
        """
        Init method. 
        Params:
            architecture_class: nn.Module class, here either FullNetwork or ReducedNetwork.
            training_temperature: float, temperature to use when training.
            teacher_model: TeacherModel object, the model used to generate the soft labels.
        """
        super().__init__()
        self.network = architecture_class() # initialize the network to use
        self.training_temperature = training_temperature
        self.teacher_model = teacher_model
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.loss_module = nn.CrossEntropyLoss() # for teacher just use normal cross entropy
        self.cur_iteration_loss = []
        self.epoch_losses = []

    def forward(self, imgs, temperature): # have to explicitly pass temperature
        return self.network(imgs) / temperature

    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [{"scheduler":scheduler, "interval":"epoch"}]
    
        
    def training_step(self, batch, batch_idx):
        imgs, labels = batch # labels not used
        with torch.no_grad():
            teacher_logits = self.teacher_model(imgs, temperature=T) # soft label ALWAYS generated with temperature T
        student_logits = self(imgs, temperature=self.training_temperature) # training temperature can be different
        soft_targets = F.softmax(teacher_logits, dim=-1)
        loss = self.loss_module(student_logits, soft_targets)
        self.cur_iteration_loss.append(loss.item())
        self.log(f"train_loss", loss, prog_bar=True)
        return loss 
    
    def on_train_epoch_end(self):
        self.epoch_losses.append(sum(self.cur_iteration_loss)/len(self.cur_iteration_loss))
        self.cur_iteration_loss = []

############################################### STRING PARSING FUNCTIONS ###############################################
def parse_teacher_str(str):
    """Converts strings of form '{F,R}T_temp{1,T}' to teacher_architecture_class and teacher_training_temperature.
    """
    # split string and rough check for validity
    first, second = str.split("_")
    assert first[1] == "T" and first[0] in ["F","R"]
    assert second[0:4] == "temp" and second[-1] in ["1", "T"]
    # get values
    teacher_architecture_class = FullNetwork if first[0] == "F" else ReducedNetwork
    teacher_training_temperature = 1 if second[-1] == "1" else T
    return teacher_architecture_class, teacher_training_temperature

def parse_student_str(str):
    """Converts strings of form '{F,R}S_temp{1,T}_from_{F,R}T_temp{1,T}' to student_architecture_class, 
    student_training_temperature, teacher_architecture_class, and teacher_training_temperature.
    """
    # split string and rough check for validity
    first, second, third, fourth, fifth = str.split("_")
    assert first[1] == "S" and first[0] in ["F","R"]
    assert second[0:4] == "temp" and second[-1] in ["1", "T"]
    assert third == "from"
    assert fourth[1] == "T" and fourth[0] in ["F","R"]
    assert fifth[0:4] == "temp" and fifth[-1] in ["1", "T"]
    # get values
    student_architecture_class = FullNetwork if first[0] == "F" else ReducedNetwork
    student_training_temperature = 1 if second[-1] == "1" else T
    teacher_architecture_class = FullNetwork if fourth[0] == "F" else ReducedNetwork
    teacher_training_temperature = 1 if fifth[-1] == "1" else T
    return student_architecture_class, student_training_temperature, teacher_architecture_class, teacher_training_temperature

############################################## WEIGHT RECOVERY FUNCTIONS ##############################################
def recover_trained_weights(str, temp=T):
    """Given either teacher or student string, recover the corresponding pytorch module.

    NOTE: Returns the PYTORCH module (of class FullNetwork or ReducedNetwork), not the Lightning model!

    Args:
        str: a str that is either in teacher or student format.
        temp: int, get model of temp.
    Returns:
        The trained model as either FullNetwork or ReducedNetwork.
    """
    num_tokens = len(str.split("_"))
    assert num_tokens in [2,5]

    # get path to model checkpoint
    ckpt_path = os.path.join(MODELS_PATH, f"{str}_T{temp}.ckpt")

    # determine if str is in teacher or student format and get lightning model
    if num_tokens == 2:
        # must be teacher model
        architecture, temp = parse_teacher_str(str)
        l_model = TeacherModel.load_from_checkpoint(ckpt_path, architecture_class=architecture,
                                                    training_temperature=temp)
    else:
        # must be student model
        architecture, temp, t_architecture, t_temp = parse_student_str(str)
        dummy_teacher = TeacherModel(t_architecture, t_temp)
        l_model = StudentModel.load_from_checkpoint(ckpt_path, architecture_class=architecture,
                                                    training_temperature=temp, teacher_model=dummy_teacher)

    # return the pytorch module inside the lightning model
    return l_model.network.eval()

################################################ DATA HELPER FUNCTIONS ################################################

class DataTOY(Dataset):
    """Wrapper for DataTOY.csv"""
    
    def __init__(self, path):
        self.df = pd.read_csv(path, index_col=[0])
        self.data = self.df.to_numpy().astype(np.float32)
        self.x = torch.from_numpy(self.data[:,0:2])
        self.y = self.data[:,2].astype(int)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return len(self.data)

def get_TOY_loaders(batch_size):
    """
    Read the TOY dataset as pytorch DataLoaders.
    Params:
        batch_size: int, batch size to use for the DataLoaders.
    Returns:
        Pytorch DataLoaders for the training set (1200), no testing set.
    """
    # get pytorch dataset
    trainset = DataTOY(os.path.join(DATA_PATH, "DataTOY.csv"))

    # convert into pytorch dataloaders and return
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader