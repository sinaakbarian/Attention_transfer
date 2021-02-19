from torch import nn
import torch
from torchvision import  models

# This fine include ResNet18NN, ResNet34NN, ResNet50NN

#-----------------------------------------------------  Resnet18

class ResNet18NN(nn.Module):
    def __init__(self, encoded_image_size=14):
        # def __init__(self):
        super(ResNet18NN, self).__init__()
        self.encoded_image_size = encoded_image_size
        # This code returns a model consisting of all layers of resnet50 bar the last 1, Fully connected layers, softmax
        resnet = models.resnet18(pretrained=True)

        modules = list(resnet.children())[:-1]  # Returns an iterator over immediate children modules
        #  It is important to note that children() returns "immediate" modules, which means if last module of your network  is a sequential, it will return whole sequential
        self.resnet = nn.Sequential(*modules)  # build a sequential nn using the list of modules defined above.

        # remove following 2 lines if you do not need encoded_image_size

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc1 = nn.Linear(512 * 14 * 14, 256)

        # self.fc1 = nn.Linear(2048 , 1024)
        # self.fc2 = nn.Linear(1024, 14)
        self.fc2 = nn.Sequential(nn.Linear(256, 14), nn.Sigmoid())
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images

        We do add an AdaptiveAvgPool2d() layer to resize the encoding to a fixed size. This makes it possible
        to feed images of variable size to the Encoder. (We did, however, resize our input images to 256, 256
        because we had to store them together as a single tensor.)
        """

        #  out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        """ resNet101, encoded images with 2048 learned channel (2048*3*14*14),  encoded_image_size=14 """
        # out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # if you want to change the order of the output tensor you can also use above line

        out = self.resnet(images)  # (batch_size, 2048, :7=image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        out = out.view(-1, 512 * 14 * 14)
        # out = out.view(-1,2048)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    """
    Transfer Learning. This is when you borrow from an existing model by using parts of it in a new model. 
    This is almost always better than training a new model from scratch (i.e., knowing nothing).
    As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand.

    If fine-tuning, only fine-tune convolutional blocks 2 through 4 ecause the first convolutional block would have usually learned
    b something very fundamental to image processing, such as detecting lines, edges, curves, etc. We don't mess with the foundations
    """

    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


#-------------------------------------------- ResNet34

class ResNet34NN(nn.Module):
    def __init__(self, encoded_image_size=14):
        # def __init__(self):
        super(ResNet34NN, self).__init__()
        self.encoded_image_size = encoded_image_size
        # This code returns a model consisting of all layers of resnet50 bar the last 1, Fully connected layers, softmax
        resnet = models.resnet34(pretrained=True)

        modules = list(resnet.children())[:-1]  # Returns an iterator over immediate children modules
        #  It is important to note that children() returns "immediate" modules, which means if last module of your network  is a sequential, it will return whole sequential
        self.resnet = nn.Sequential(*modules)  # build a sequential nn using the list of modules defined above.

        # remove following 2 lines if you do not need encoded_image_size

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fc1 = nn.Linear(512 * 14 * 14, 256)

        # self.fc1 = nn.Linear(2048 , 1024)
        # self.fc2 = nn.Linear(1024, 14)
        self.fc2 = nn.Sequential(nn.Linear(256, 14), nn.Sigmoid())
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images

        We do add an AdaptiveAvgPool2d() layer to resize the encoding to a fixed size. This makes it possible
        to feed images of variable size to the Encoder. (We did, however, resize our input images to 256, 256
        because we had to store them together as a single tensor.)
        """

        #  out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        """ resNet101, encoded images with 2048 learned channel (2048*3*14*14),  encoded_image_size=14 """
        # out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # if you want to change the order of the output tensor you can also use above line

        out = self.resnet(images)  # (batch_size, 2048, :7=image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        out = out.view(-1, 512 * 14 * 14)
        # out = out.view(-1,2048)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    """
    Transfer Learning. This is when you borrow from an existing model by using parts of it in a new model. 
    This is almost always better than training a new model from scratch (i.e., knowing nothing).
    As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand.

    If fine-tuning, only fine-tune convolutional blocks 2 through 4 ecause the first convolutional block would have usually learned
    b something very fundamental to image processing, such as detecting lines, edges, curves, etc. We don't mess with the foundations
    """

    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


#----------------------------------------  ResNet50NN
class ResNet50NN(nn.Module):
    def __init__(self, encoded_image_size=14):
   # def __init__(self):
        super(ResNet50NN, self).__init__()
        self.encoded_image_size = encoded_image_size
        #This code returns a model consisting of all layers of resnet50 bar the last 1, Fully connected layers, softmax 
        resnet = models.resnet50(pretrained=True)
        
        
        modules = list(resnet.children())[:-1] #Returns an iterator over immediate children modules
        #  It is important to note that children() returns "immediate" modules, which means if last module of your network  is a sequential, it will return whole sequential
        self.resnet = nn.Sequential(*modules) # build a sequential nn using the list of modules defined above.
      
       # remove following 2 lines if you do not need encoded_image_size
        

        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) 
        self.fc1 = nn.Linear(2048*14*14 , 1024)
    
        #self.fc1 = nn.Linear(2048 , 1024)
       # self.fc2 = nn.Linear(1024, 14)
        self.fc2 = nn.Sequential(nn.Linear(1024, 14), nn.Sigmoid())
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        
        We do add an AdaptiveAvgPool2d() layer to resize the encoding to a fixed size. This makes it possible
        to feed images of variable size to the Encoder. (We did, however, resize our input images to 256, 256
        because we had to store them together as a single tensor.)
        """
        
      #  out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        """ resNet101, encoded images with 2048 learned channel (2048*3*14*14),  encoded_image_size=14 """
        #out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # if you want to change the order of the output tensor you can also use above line
        
        out = self.resnet(images)  # (batch_size, 2048, :7=image_size/32, image_size/32)
        out = self.adaptive_pool(out)
        out = out.view(-1,2048*14*14)
        #out = out.view(-1,2048)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

    """
    Transfer Learning. This is when you borrow from an existing model by using parts of it in a new model. 
    This is almost always better than training a new model from scratch (i.e., knowing nothing).
    As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand.
        
    If fine-tuning, only fine-tune convolutional blocks 2 through 4 ecause the first convolutional block would have usually learned
    b something very fundamental to image processing, such as detecting lines, edges, curves, etc. We don't mess with the foundations
    """
    def fine_tune(self, fine_tune=True):

        for p in self.resnet.parameters():
            p.requires_grad = False
            
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

