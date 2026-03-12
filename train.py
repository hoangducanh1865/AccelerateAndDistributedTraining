import os
import argparse
import torch.nn as nn
from accelerate import Accelerator 
from utils import LocalLogger
from torchmetrics import Accuracy
from torchvision.models import resnet50
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

### Argument Parsing ###
parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")
parser.add_argument("--working_directory",
                    help="Where checkpoints and logs are stored, inside a folder labeled by the experiment name",
                    required=True,
                    type=str)
parser.add_argument("--experiment_name",
                    help="Name of experiment being lanched",
                    type=str)
parser.add_argument("--gradient_accumulation_steps",
                    help="Number of gradient accumulation steps for training",
                    default=1,
                    type=int)
parser.add_argument("--epochs",
                    help="Number of epochs to train",
                    default=90,
                    type=int)
parser.add_argument("--batch_size",
                    help="Effective batch_size. If split_batches is False, batch size is multiplied by number of GPUs utilized",
                    default=64,
                    type=int)
parser.add_argument("--learning_rate",
                    help="Start learning rate for StepLR",
                    default=0.1,
                    type=float)
parser.add_argument("--num_classes",
                    help="How many classes is out network/model predicting?",
                    default=1000,
                    type=int)
parser.add_argument("--img_size",
                    help="Width and height of images passed to the model",
                    default=224,
                    type=int)
parser.add_argument("--path_to_data",
                    help="Path to Imagenet root folder which should contain train/ and validation/ folders",
                    required=True,
                    type=str)
parser.add_argument("--num_workers",
                    help="Number of workers for DataLoader",
                    default=32,
                    type=int)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
parser.add_argument("--arg",
                    help="help",
                    type=str)
args = parser.parse_args()

### Accelerator Setup ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb")
experiment_config = {
    "epochs": args.epochs,
    "effective_batch_size": args.batch_size * accelerator.num_processes,
    "learning_rate": args.learning_rate
}
accelerator.init_trackers(project_name=args.experiment_name,
                          config=experiment_config)

### Init Logger ###
local_logger = LocalLogger(path_to_log_folder=path_to_experiment)

### Define Accuracy Metric ###
accuracy_fn = Accuracy(task="multiclass", num_classes=args.num_classes).to(accelerator.device)

### Load Model ###
model = resnet50()
if args.num_classes != 1000: 
    model.fc = nn.Linear(2048, args.num_classes) # Replace prediction head with number of classes
    # Q

### Transforms & Dataset ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
) # Q

# Load Dataset
path_to_train_data = os.path.join(args.path_to_data, "train")
path_to_valid_data = os.path.join(args.path_to_data, "validation")
trainset = datasets.ImageFolder(root=path_to_train_data, transform=train_transforms)
testset = datasets.ImageFolder(root=path_to_valid_data, transform=test_transform)

mini_batchsize = args.batch_size // args.gradient_accumulation_steps
trainloader = DataLoader(dataset=trainset, batch_size=mini_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True) # Q: pin_memory=True?
test_loader = DataLoader(dataset=testset, batch_size=mini_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True) # Q: pin_memory=True?

### Define Loss Function ###
loss_fn = nn.CrossEntropyLoss()

### Selective Weight Decay, Optimizer & Scheduler ###


### Prepare Everything ###


### Training Loop ###


### Metrics Gathering & Logging ###


### Checkpointing & Resuming ### 