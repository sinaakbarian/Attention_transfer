from dataset import *
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def AUC_for_log(model, test_df, path_image, device):
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

    print("path_image", path_image)
    print("test_df",test_df)
    BATCH_SIZE = 32
    workers = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_test = MIMICCXRDataset(test_df, path_image=path_image, transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize]))
    test_loader = torch.utils.data.DataLoader(dataset_test, BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

    size = len(test_df)
    print("Test _df size :", size)



    # criterion = nn.BCELoss().to(device)
    model = model.to(device)
    # to find this thresold, first we get the precision and recall withoit this, from there we calculate f1 score, using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation are used to calculate our binary output.

    PRED_LABEL = PRED_LABEL = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
            'Edema', 'Consolidation', 'Pneumonia','Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other',
            'Fracture',  'Support Devices']

    
    # create empty dfs
    pred_df = pd.DataFrame(columns=["Path"])
    bi_pred_df = pd.DataFrame(columns=["Path"])
    true_df = pd.DataFrame(columns=["Path"])           
    loader = test_loader
    TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])
    for i, data in enumerate(loader):
            inputs, labels, item = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                probs = outputs.cpu().data.numpy()
            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                truerow = {}

                truerow["Path"] = item[j]
                thisrow["Path"] = item[j]
                

                # iterate over each entry in prediction vector; each corresponds to
                # individual label
                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)
           
            if (i % 200 == 0):
                print(str(i * BATCH_SIZE))


 
    for column in true_df:
            if column not in PRED_LABEL:
                    continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            thisrow['auc'] = np.nan
            thisrow['auprc'] = np.nan
            

            thisrow['auc'] = sklm.roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())
            thisrow['auprc'] = sklm.average_precision_score(actual.to_numpy().astype(int), pred.to_numpy())
            TestEval_df = TestEval_df.append(thisrow, ignore_index=True)

    print("AUC ave:", TestEval_df['auc'].sum() / 14.0)

    print("done")

    return TestEval_df['auc'].sum()/14.0 # , bi_pred_df , Eval_bi_df

