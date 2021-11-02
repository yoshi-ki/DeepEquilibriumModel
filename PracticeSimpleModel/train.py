import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from solver import anderson
from modelClasses import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################
# create model #
################
torch.manual_seed(0)
chan = 48
f = ResNetLayer(chan, 64, kernel_size=3)
model = nn.Sequential(nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1),
                      nn.BatchNorm2d(chan),
                      DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
                      nn.BatchNorm2d(chan),
                      nn.AvgPool2d(8,8),
                      nn.Flatten(),
                      nn.Linear(chan*4*4,10)).to(device)

#######################
# CIFAR10 data loader #
#######################
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cifar10_train = datasets.CIFAR10(".", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar10_train, batch_size = 100, shuffle=True, num_workers=8)
test_loader = DataLoader(cifar10_test, batch_size = 100, shuffle=False, num_workers=8)

########################################
# standard training or evaluation loop #
########################################
def epoch(loader, model, opt=None, lr_scheduler=None):
  total_loss, total_err = 0.,0.
  model.eval() if opt is None else model.train()
  for X,y in loader:
    X,y = X.to(device), y.to(device)
    yp = model(X)
    loss = nn.CrossEntropyLoss()(yp,y)
    if opt:
      opt.zero_grad()
      loss.backward()
      opt.step()
      lr_scheduler.step()

    total_err += (yp.max(dim=1)[1] != y).sum().item()
    total_loss += loss.item() * X.shape[0]

  return total_err / len(loader.dataset), total_loss / len(loader.dataset)

########################
# train and evaluation #
########################
import torch.optim as optim
opt = optim.Adam(model.parameters(), lr=1e-3)
print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

max_epochs = 2
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

train_errs = np.array([])
train_losses = np.array([])
test_errs = np.array([])
test_losses = np.array([])

for i in range(max_epochs):
  print("epoch:", i)
  train_err, train_loss = epoch(train_loader, model, opt, scheduler)
  print("train:", train_err, train_loss)
  test_err, test_loss = epoch(test_loader, model)
  print("test:", test_err, test_loss)

  train_errs = np.append(train_errs,train_err)
  train_losses = np.append(train_losses, train_loss)
  test_errs = np.append(test_errs, test_err)
  test_losses = np.append(test_losses, test_loss)

################
# plot results #
################
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.arange(max_epochs),train_errs, label="train")
plt.plot(np.arange(max_epochs), test_errs, label="test")
plt.legend()
plt.savefig("result-error-cifar10.png")

plt.figure()
plt.plot(np.arange(max_epochs), train_losses, label="train")
plt.plot(np.arange(max_epochs), test_losses, label="test")
plt.legend()
plt.savefig("result-loss-cifar10.png")