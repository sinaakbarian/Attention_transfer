
import torch
from utils import *
import numpy as np
from torch import nn
from torchvision import  models
#from evaluation import *
activation_t = {}
activation_s = {}
def get_activation_t(name):
    def hook(model, input, output):
        activation_t[name] = output
    return hook

def get_activation_s(name):
    def hook(model, input, output):
        activation_s[name] = output
    return hook

def loss_function(get_activation_t,get_activation_s,criterion,outputs, labels,device):

    BCE = criterion(outputs,labels)
    MSEL=nn.MSELoss().to(device)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    Loss_attention_all = []
    for (name, val) in activation_t.items():
        #print(name)
        teacher=activation_t[str(name)]
        student=activation_s[str(name)]
        normalized_t=torch.norm(teacher,p=2,dim=(2,3))
        normalized_s=torch.norm(student,p=2,dim=(2,3))

        l_t=teacher/(normalized_t.view(normalized_t.shape[0],normalized_t.shape[1],1,1))
        l_s=student/(normalized_s.view(normalized_s.shape[0],normalized_s.shape[1],1,1))
        #new
        #l_t=torch.mean(torch.abs(l_t),dim=1)
        #l_s=torch.mean(torch.abs(l_s),dim=1)
        #average_t=torch.mean(torch.abs(teacher),dim=1)
        #average_s=torch.mean(torch.abs(student),dim=1)
        #normalized_s=torch.sqrt(torch.sum(torch.sum(torch.pow(average_s, 2),dim=-1),dim=-1))
        #normalized_t=torch.sqrt(torch.sum(torch.sum(torch.pow(average_t, 2),dim=-1),dim=-1))
        #l_t=average_t/(normalized_t.view(normalized_t.shape[0],1,1,1))
        #l_s=average_s/(normalized_s.view(normalized_s.shape[0],1,1,1))
        #Loss_attention=torch.sqrt(torch.sum(torch.sum(torch.sum(torch.pow(l_s-l_t, 2),dim=-1),dim=-1),dim=-1))
        Loss_attention=torch.norm(l_t-l_s,p=2)
        #Loss_attention=torch.mean(Loss_attention)
        
        Loss_attention_all.append(Loss_attention)
    
    Loss_attention_all = torch.stack(Loss_attention_all)
    
    Loss_attention_all=torch.mean(Loss_attention_all)
    print("Loss Attention:",Loss_attention_all)
    print("Loss BCE: ",BCE)
    #Attention transfer for mimic
    '''
    print("all",BCE+1/4*Loss_attention_all)
    return BCE+1/4*Loss_attention_all
    '''
    #Attention transfer for Imagenet
    
    print("all",BCE+1/4*Loss_attention_all)
    return BCE+Loss_attention_all/1000
    


def BatchIterator(model, phase,
        Data_loader,
        criterion,
        optimizer,
        device):
    #___________________Initial teacher's parameters
    #mimic teacher
    '''
    CheckPointDataforM = torch.load('/scratch/gobi2/sinaakb/CheXpert/M-Model/results47/checkpoint')
    modelM_teacher = CheckPointDataforM['model']
    '''
    #imagenet teacher
    modelM_teacher = models.densenet121(pretrained=True)
    
    modelM_teacher=modelM_teacher.to(device)
    for param in modelM_teacher.parameters():
        param.requires_grad=False
    modelM_teacher.eval()
    '''
    for name, module in modelM_teacher.named_modules():
        print(name)
    '''
    #print("loading teacher")
    modelM_teacher.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation_t('1'))
    #modelM_teacher.features.denseblock3.denselayer16.conv2.register_forward_hook(get_activation_t('2'))
    #modelM_teacher.features.denseblock3.denselayer8.conv2.register_forward_hook(get_activation_t('3'))
    #modelM_teacher.features.denseblock3.denselayer2.conv2.register_forward_hook(get_activation_t('4'))
    #modelM_teacher.features.denseblock2.denselayer7.conv2.register_forward_hook(get_activation_t('5'))
    #modelM_teacher.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation_t('6'))
    #modelM_teacher.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation_t('7'))



    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 2000
    running_loss = 0.0

    outs = []
    gts = []

    for i, data in enumerate(Data_loader):

        imgs, labels, _ = data
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            #print("loading student")
            # You should use module to use multiple gpus
            model.features.denseblock4.denselayer16.conv2.register_forward_hook(get_activation_s('1'))
            #model.features.denseblock3.denselayer16.conv2.register_forward_hook(get_activation_s('2'))
            #model.features.denseblock3.denselayer8.conv2.register_forward_hook(get_activation_s('3'))
            #model.features.denseblock3.denselayer2.conv2.register_forward_hook(get_activation_s('4'))
            #model.features.denseblock2.denselayer7.conv2.register_forward_hook(get_activation_s('5'))
            #model.features.denseblock1.denselayer6.conv2.register_forward_hook(get_activation_s('6'))
            #model.features.denseblock1.denselayer1.conv2.register_forward_hook(get_activation_s('7'))
            #print(" done loading student")
            outputs = model(imgs)
            teacher_output=modelM_teacher(imgs)

        else:

            for label in labels.cpu().numpy().tolist():
                gts.append(label)

            model.eval()
            with torch.no_grad():
                outputs = model(imgs)
                teacher_output=modelM_teacher(imgs)
               # out = torch.sigmoid(outputs).data.cpu().numpy()
               # outs.extend(out)
            # outs = np.array(outs)
            # gts = np.array(gts)
           # evaluation_items(gts, outs)
        #print("start loss")

        loss = loss_function(get_activation_t,get_activation_s,criterion,outputs, labels,device)
        #print("end loss")
        if phase == 'train':

            loss.backward(retain_graph=True)
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights
            #torch.cuda.empty_cache()
        running_loss += loss.item() * batch_size

        if (i % 500 == 0):
            print(str(i * batch_size))





    return running_loss
