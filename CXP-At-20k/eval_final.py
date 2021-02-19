from dataset import *
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

def eval(model, test_df, path_image, device):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model
    Args:

        model: densenet-121 from torchvision previously fine tuned to training data
        test_df : dataframe csv file
        PATH_TO_IMAGES:
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    BATCH_SIZE = 32
    workers = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_test = MIMICCXRDataset(test_df, path_image=path_image, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))

    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)

    size = len(test_df)
    print("Test _df size :", size)

    Experiment_name=[]
    accuracy=[]
    precision=[]
    recall=[]

    f1score=[]
    AUC=[]
    average_precision=[]

    # criterion = nn.BCELoss().to(device)
    model = model.to(device)
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.
    All_true=np.zeros((1, 14))
    All_predict=np.zeros((1, 14))
    for i, data in enumerate(test_loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels = labels.cpu().data.numpy()
            
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                prediction=outputs.cpu().numpy()
                All_true=np.concatenate((All_true,true_labels))
                All_predict=np.concatenate((All_predict,prediction))
               # out = torch.sigmoid(outputs).data.cpu().numpy()
                
    All_true = np.delete(All_true, (0), axis=0)
    All_predict = np.delete(All_predict, (0), axis=0)
    AUC_all=0
    accuracy_all=0
    f1_score_all=0
    precision_score_all=0
    average_precision_all=0
    recall_all=0
    for labels_n in range(All_true.shape[1]):
                    print(labels_n)
                    
                    accuracy_epoch=sklm.accuracy_score(All_true[:,labels_n].reshape(All_true.shape[0]), np.around(All_predict[:,labels_n].reshape(All_true.shape[0])))
                    f1_score_epoch=sklm.f1_score(All_true[:,labels_n].reshape(All_true.shape[0]), np.around(All_predict[:,labels_n].reshape(All_true.shape[0])))
                    precision_score_epoch=sklm.precision_score(All_true[:,labels_n].reshape(All_true.shape[0]), np.around(All_predict[:,labels_n].reshape(All_true.shape[0])))
                    average_precision_epoch=sklm.average_precision_score(All_true[:,labels_n].reshape(All_true.shape[0]),All_predict[:,labels_n].reshape(All_true.shape[0]))
                    recall_epoch=sklm.recall_score(All_true[:,labels_n].reshape(All_true.shape[0]), np.around(All_predict[:,labels_n].reshape(All_true.shape[0])))
                    AUC_epoch_each_label=sklm.roc_auc_score(All_true[:,labels_n].reshape(All_true.shape[0]), All_predict[:,labels_n].reshape(All_true.shape[0]))

                    accuracy_all+=(accuracy_epoch)
                    f1_score_all+=(f1_score_epoch)
                    precision_score_all+=(precision_score_epoch)
                    average_precision_all+=(average_precision_epoch)
                    recall_all+=(recall_epoch)
                    AUC_all+=(AUC_epoch_each_label)

    
 


    Experiment_name.append("Normal_100k")
    accuracy.append(np.round((accuracy_all/true_labels.shape[1])*100,2))
    precision.append(np.round((precision_score_all/true_labels.shape[1])*100,2))
    recall.append(np.round((recall_all/true_labels.shape[1])*100,2))
    f1score.append(np.round((f1_score_all/true_labels.shape[1])*100,2))
    AUC.append(np.round((AUC_all/true_labels.shape[1])*100,2))
    average_precision.append(np.round((average_precision_all/true_labels.shape[1])*100,2))
    df = pd.DataFrame({'Experiment_name': Experiment_name,'accuracy': accuracy,'precision': precision,'recall': recall,'f1score': f1score,'AUC': AUC,'average_precision': average_precision})
    df.to_csv("beta_5_reintiallized_AT-alldata-77.csv",index=False)

path_image = "/scratch/gobi2/sinaakb/CheXpert"

CheckPointData = torch.load('/scratch/gobi2/sinaakb/CheXpert/CXP-Model/results40/checkpoint')
model = CheckPointData['model']
weights=torch.load('results/checkpoint-all-reintiallized_AT-mimic-alldata---7--59')
model.load_state_dict(weights['model'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df ="/scratch/gobi2/sinaakb/CheXpert/split/new_test.csv"
test_df = pd.read_csv(test_df)
eval(model, test_df, path_image, device)
