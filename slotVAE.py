# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from tqdm import tqdm
import os

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4) * 16, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part  # Latent vectors mu and sigma
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        self.fc35 = nn.Linear(l_dim, z_dim)  # location
        self.fc36 = nn.Linear(l_dim, z_dim)
        self.fc37 = nn.Linear(sc_dim, z_dim)  # scale
        self.fc38 = nn.Linear(sc_dim, z_dim)
        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc4l = nn.Linear(z_dim, l_dim)  # location
        self.fc4sc = nn.Linear(z_dim, sc_dim)  # scale

        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        self.fc8 = nn.Linear(h_dim2, h_dim2) #skip conection

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()

        # map scalars
        self.shape_scale = 1 #1.9
        self.color_scale = 1 #2

    def encoder(self, x, l):
        l = l.view(-1,l_dim)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
        h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), self.fc35(l), self.fc36(l) # mu, log_var

    def sampling_location(self, mu, log_var):
        std = (0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_location(self, z_shape, z_color, z_location):
        h = self.fc4l(z_location)
        return torch.sigmoid(h).view(-1,2,retina_size)

    def decoder_scale(self, z_shape, z_color, z_scale):
        h = self.fc4sc(z_scale)
        return torch.sigmoid(h).view(-1,10)

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color, z_location, hskip=0):
        h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_skip_cropped(self, z_shape, z_color, z_location, hskip):
        h = F.relu(hskip)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        if type(x) == list or type(x) == tuple:    #passing in a cropped+ location as input
            l = x[2].cuda()
            #sc = x[3].cuda()
            x = x[1].cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)
        else:  #passing in just cropped image
            x = x.cuda()
            #sc = torch.zeros(x.size()[0], sc_dim).cuda()
            l = torch.zeros(x.size()[0], self.l_dim).cuda()
            mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, mu_scale, log_var_scale, hskip = self.encoder(x, l)

        #what maps are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()

        if ('location' in keepgrad):
            z_location = self.sampling_location(mu_location, log_var_location)
        else:
            z_location = self.sampling_location(mu_location, log_var_location).detach()

        if ('skip' in keepgrad):
            hskip = hskip
        else:
            hskip = hskip.detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape,z_color, z_location, hskip)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, z_location, z_scale=0)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(0, 0, 0, hskip)
        elif (whichdecode == 'skip_retinal'):
            output = self.decoder_skip_retinal(0, 0, z_location, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0)
        elif (whichdecode == 'location'):
            output = self.decoder_location(0, 0, z_location)
        elif (whichdecode == 'scale'):
            output = self.decoder_scale(0, 0, 0, z_scale=0)
        return output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale


#######what optimier to use:
# learning rate = 0.0001
#optimizer = torch.optim.SGD(vae.parameters(), lr=0.0001, momentum = 0.9)
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

vae.cuda()

######the loss functions
#Pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x[0].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3, retina_size, retina_size), x.view(-1, 3, retina_size, retina_size), reduction='sum')
    return BCE

#pixelwise loss for just the cropped image
def loss_function_crop(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x.clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), x.view(-1, imgsize * imgsize * 3), reduction='sum')
    return BCE


# loss for shape in a cropped image
def loss_function_shape(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make grayscale reconstruction
    gray_x = x.view(-1, 3, imgsize, imgsize).mean(1)
    gray_x = torch.stack([gray_x, gray_x, gray_x], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), gray_x.view(-1,imgsize * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

#loss for just color in a cropped image
def loss_function_color(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(-1, 3 * imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, imgsize * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just location
def loss_function_location(recon_x, x, mu, log_var):
    x = x[2].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,2,retina_size), x.view(-1,2,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

#loss for just scale
def loss_function_scale(recon_x, x, mu, log_var):
    x = x[3].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1,retina_size,retina_size), x.view(-1,retina_size,retina_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def test_loss(test_data, whichdecode = []):
    loss_dict = {}

    for decoder in whichdecode:
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale = vae(test_data, decoder)
        
        if decoder == 'retinal':
            loss = loss_function(recon_batch, test_data, mu_shape, log_var_shape, mu_color, log_var_color)
        
        elif decoder == 'cropped':
            loss = loss_function_crop(recon_batch, test_data[1], mu_shape, log_var_shape, mu_color, log_var_color)
        
        loss_dict[decoder] = loss.item()

    return loss_dict

def train(epoch, train_loader_noSkip, emnist_skip, fmnist_skip, test_loader, sample_loader, return_loss = False, seen_labels = {}):
    vae.train()
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) # the latent space is trained on EMNIST, MNIST, and f-MNIST
    m = 5 # number of seperate training decoders used
    if fmnist_skip != None:
        m=6
        #dataiter_emnist_skip= iter(emnist_skip) # The skip connection is trained on pairs from EMNIST, MNIST, and f-MNIST composed on top of each other
        dataiter_fmnist_skip= iter(fmnist_skip)
    test_iter = iter(test_loader)
    sample_iter = iter(sample_loader)
    count = 0
    max_iter = 200
    loader=tqdm(train_loader_noSkip, total = max_iter)

    retinal_loss_train, cropped_loss_train = 0, 0 # loss metrics returned to training.py
    
    if epoch > 51: # increase the number of times retinal/location is trained
        m = 7

    for i,j in enumerate(loader):
        count += 1
        data_noSkip, batch_labels = next(dataiter_noSkip)
        if count % m == 4:
            #r = random.randint(0,1)
            #if r == 1:
             #   data_skip = dataiter_emnist_skip.next()
            #else:
            data_skip = next(dataiter_fmnist_skip)
    
        data = data_noSkip
        
        optimizer.zero_grad()
        
        if count% m == 0:
            whichdecode_use = 'retinal'
            keepgrad = []

        elif count% m == 1:
            whichdecode_use = 'color'
            keepgrad = ['color']

        elif count% m == 2:
            whichdecode_use = 'location'
            keepgrad = ['location']

        elif count% m == 3:
            if epoch <= 20:
                whichdecode_use = 'location'
                keepgrad = ['location']
            else:
                whichdecode_use = 'retinal'
                keepgrad = [] #all except skip connection

        elif count% m == 4:
            whichdecode_use = 'cropped'
            keepgrad = ['shape', 'color']

        elif count% m == 5:
            r = random.randint(0,1)
            if r == 1:
                data = data_skip[0]
            else:
                data = data[1]
            whichdecode_use = 'skip_cropped'
            keepgrad = ['skip']
        
        else:
            whichdecode_use = 'retinal'
            keepgrad = []

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location, mu_scale, log_var_scale = vae(data, whichdecode_use, keepgrad)
            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)
            loss.backward()

        elif whichdecode_use == 'color': # color
            loss = loss_function_color(recon_batch, data, mu_color, log_var_color)
            loss.backward()

        elif whichdecode_use == 'location': # location
            loss = loss_function_location(recon_batch, data, mu_location, log_var_location)
            loss.backward()

        elif whichdecode_use == 'retinal': # retinal
            loss = loss_function(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()
            retinal_loss_train = loss.item()

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data[1], mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()
            cropped_loss_train = loss.item()

        elif whichdecode_use == 'skip_cropped': # skip training
            loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            loss.backward()

        elif whichdecode_use == 'scale': # scale training
            loss = loss_function_crop(recon_batch, data, mu_scale, log_var_scale)
            loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description((f'epoch: {epoch}; mse: {loss.item():.5f};'))
        seen_labels = update_seen_labels(batch_labels,seen_labels)
        if count % (0.8*max_iter) == 0:
            data, labels = next(sample_iter)
            progress_out(data, epoch, count)
        #elif count % 500 == 0: not for RED GREEN
         #   data = data_noSkip[0][1] + data_skip[0]
          #  progress_out(data, epoch, count, skip= True)
        
        if i == max_iter +1:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))
    
    if return_loss is True:
        # get test losses for cropped and retinal
        test_data = next(test_iter)
        test_data = test_data[0]

        test_loss_dict = test_loss(test_data, ['retinal', 'cropped'])
    
        return [retinal_loss_train, test_loss_dict['retinal'], cropped_loss_train, test_loss_dict['cropped']], seen_labels