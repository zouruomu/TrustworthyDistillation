
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import torchvision
import imageio
import os

###################################################### CONSTANTS ######################################################
T = 20 # temperature, constant for all
DATA_PATH = "../Data/"
MODELS_PATH = "../Models/"
FIGURES_PATH = "../GeneratedFigures/"
NUM_CLASSES = 58
RANDOM_SEED = 42

# torch image transform normalization parameters
NORMALIZE_MEAN = 0.5 # same for all channels
NORMALIZE_STD = 0.5 # same for all channels

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
    Module implementing baseline CNN architecture for TSRD.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        """
        Init method.
        Params:
            num_classes: int, NUM_CLASSES.
            dropout: float, 0.5 in accordance with DD paper's baseline.
        """
        super().__init__()

        # conv block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # conv block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # conv block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(64*4*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """Forward method."""
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ReducedNetwork(nn.Module):
    """
    Reduced version of FullNetwork to distill knowledge into.
    NOTE: Original DD paper used teacher/student models that are both FullNetworks.
          I will do the same, and additionally try a ReducedNetwork student.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        """
        Init method.
        Params:
            num_classes: int, NUM_CLASSES.
            dropout: float, 0.5 in accordance with DD paper's baseline.
        """
        super().__init__()

        # conv block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # conv block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # conv block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(32*4*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward method."""
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
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
        optimizer = optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=0.2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
        optimizer = optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=0.2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
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
def recover_trained_weights(str):
    """Given either teacher or student string, recover the corresponding pytorch module.

    NOTE: Returns the PYTORCH module (of class FullNetwork or ReducedNetwork), not the Lightning model!

    Args:
        str: a str that is either in teacher or student format.
    Returns:
        The trained model as either FullNetwork or ReducedNetwork.
    """
    num_tokens = len(str.split("_"))
    assert num_tokens in [2,5]

    # get path to model checkpoint
    ckpt_path = os.path.join(MODELS_PATH, f"{str}.ckpt")

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

def get_TSRD_loaders(batch_size):
    """Read the TSRD dataset as pytorch DataLoaders.

    NOTE: Across train and test, the max image width is 491, and the max image height is 402.
    
    Params:
        batch_size: int, batch size to use for the DataLoaders.
    Returns:
        Pytorch DataLoaders for the training set (4170) and test set (1994).
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(256,256)),
            torchvision.transforms.Normalize((NORMALIZE_MEAN,NORMALIZE_MEAN,NORMALIZE_MEAN),
                                             (NORMALIZE_STD,NORMALIZE_STD,NORMALIZE_STD))
        ])

    # define loader function for DatasetFolder
    def TSRD_loader(path):
        return imageio.imread(path)

    # read train and test sets as pytorch datasets
    trainset = torchvision.datasets.DatasetFolder(root=os.path.join(DATA_PATH, "PytorchFriendlyFormat/", "Train/"),
                                                  loader=TSRD_loader, extensions=["png"], allow_empty=True,
                                                  transform=transform)
    
    testset = torchvision.datasets.DatasetFolder(root=os.path.join(DATA_PATH, "PytorchFriendlyFormat/", "Test/"),
                                                 loader=TSRD_loader, extensions=["png"], allow_empty=True,
                                                 transform=transform)
    
    # convert into pytorch dataloaders and return
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def denormalize_imgs(imgs):
    """Denormalizes images normalized with NORMALIZE_MEAN and NORMALIZE_STD."""
    return imgs*NORMALIZE_STD + NORMALIZE_MEAN
