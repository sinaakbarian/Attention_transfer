import pandas as pd
import numpy as np
from scipy import stats


def tpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1


def preprocess_CXP(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/CheXpert/map.csv")
    if 'Atelectasis' in split.columns:
        details = details.drop(columns=['No Finding',
                                        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                                        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                                        'Support Devices'])
    split = split.merge(details, left_on="Path", right_on="Path")
    split['Age'] = np.where(split['Age'].between(0, 19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20, 39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40, 59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60, 79), 79, split['Age'])
    split['Age'] = np.where(split['Age'] >= 80, 81, split['Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81, 'Male', 'Female'],
                          [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])
    return split


def preprocess_MIMIC(split):
    # total_subject_id = pd.read_csv("total_subject_id_with_gender.csv")
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/mimic-cxr-metadata-detail.csv")
    details = details.drop(columns=['dicom_id', 'study_id'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    if "subject_id" not in split.columns:
        subject_id = []
        for idx, row in split.iterrows():
            subject_id.append(row['path'].split('/')[1][1:])
        split['subject_id'] = subject_id
        split = split.sort_values("subject_id")
    if "gender" not in split.columns:
        split["subject_id"] = pd.to_numeric(split["subject_id"])
        split = split.merge(details, left_on="subject_id", right_on="subject_id")
    split = split.replace(
        [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
         '>=90'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
         'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    return split


def preprocess_NIH(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/NIH/preprocessed.csv")
    if 'Cardiomegaly' in split.columns:
        split = split.drop(columns=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                                    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                                    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'])
    split = details.merge(split, left_on='Image Index', right_on='path')
    split.drop_duplicates(subset="path", keep="first", inplace=True)
    split['Patient Age'] = np.where(split['Patient Age'].between(0, 19), 19, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(20, 39), 39, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(40, 59), 59, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(60, 79), 79, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'] >= 80, 81, split['Patient Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81],
                          [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    return split





# def calculate_cor_MIMIC(df, diseases, category, category_name):
#     print('COR for MIMIC ' + category_name + ' ===================================================')
#     df = preprocess_MIMIC(df)
#     map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
#     df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
#     result = []
#     for c in category:
#         for d in diseases:
#             pred_disease = "bi_" + d
#             gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
#             pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
#             n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
#             n_pred = df.loc[
#                      (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
#             pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
#             pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]
#
#             if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
#                 TPR = len(pred) / len(gt)
#                 n_TPR = len(n_pred) / len(n_gt)
#                 percentage = len(pi_gy) / len(pi_y)
#                 if category_name != 'Patient Gender':
#                     temp = []
#                     for c1 in category:
#                         ret = tpr(df, d, c1, category_name)
#                         if ret != -1:
#                             temp.append(ret)
#                     temp.sort()
#
#                     if len(temp) % 2 == 0:
#                         median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
#                     else:
#                         median = temp[(len(temp) // 2)]
#                     GAP = TPR - median
#                 else:
#                     GAP = TPR - n_TPR
#                 result.append([percentage, GAP])
#             else:
#                 result.append([50, 50])
#
#     result = np.array(result)
#     mask = result[:, 1] < 50
#     print("coeff for " + category_name + " is " + str(stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))


def calculate_cor_NIH_second(df, diseases, category, category_name):
    print('COR for NIH ' + category_name + '===================================================')
    df = preprocess_NIH(df)
    # result = []
    sum_corr =0
    sum_P=0
    count = 0

    for c in category:
        result = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Patient Gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                result.append([percentage, GAP])
            else:
                result.append([50, 50])
        result = np.array(result)
        mask = result[:, 1] < 50
        print("coeff for " + str(c) + " in " + category_name + " is " + str(
            stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))

        corr = stats.pearsonr(result[:, 1][mask], result[:, 0][mask])
        sum_corr = corr[0] + sum_corr
        sum_P = corr[1] + sum_P
        count = count + 1

    print("Average coeff in " + category_name + " is " + str(sum_corr / count))
    print("Average P_value in " + category_name + " is " + str(sum_P / count))



def calculate_cor_CXP_second(df, diseases, category, category_name):
    print('COR for CXP ' + category_name + ' ===================================================')
    df = preprocess_CXP(df)
    # result = []
    sum_corr =0
    sum_P=0
    count = 0
    for c in category:
        result = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Patient Gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                result.append([percentage, GAP])
            else:
                result.append([50, 50])
        result = np.array(result)
        mask = result[:, 1] < 50
        print("coeff for " + str(c) + " in " + category_name + " is " + str(
            stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))

        corr = stats.pearsonr(result[:, 1][mask], result[:, 0][mask])
        sum_corr = corr[0] + sum_corr
        sum_P = corr[1] + sum_P
        count = count + 1

    print("Average coeff in " + category_name + " is " + str(sum_corr / count))
    print("Average P_value in " + category_name + " is " + str(sum_P / count))

    # result = np.array(result)
    # mask = result[:, 1] < 50
    # print("coeff for " + category_name + " is " + str(stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))


def calculate_cor_MIMIC_second(df, diseases, category, category_name):
    print('COR for MIMIC ' + category_name + ' ===================================================')
    df = preprocess_MIMIC(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")

    sum_corr =0
    sum_P=0
    count = 0
    for c in category:
        result = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Patient Gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                result.append([percentage, GAP])
            else:
                result.append([50, 50])
        result = np.array(result)
        mask = result[:, 1] < 50
        print("coeff for " + str(c) + " in " + category_name + " is " + str(
            stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))

        corr = stats.pearsonr(result[:, 1][mask], result[:, 0][mask])
        sum_corr = corr[0] + sum_corr
        sum_P = corr[1] + sum_P
        count = count + 1

    print("Average coeff in " + category_name + " is " + str(sum_corr/count))
    print("Average P_value in " + category_name + " is " + str(sum_P/count))

    # result = np.array(result)
    # mask = result[:, 1] < 50
    # print("coeff for " + category_name + " is " + str(stats.pearsonr(result[:, 1][mask], result[:, 0][mask])))


if __name__ == "__main__":
    # ******************************  NIH data  ******************************
    # diseases_NIH = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #                 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    #                 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    # age_decile_NIH = ['60-80', '40-60', '20-40', '80-', '0-20']
    # gender_NIH = ['M', 'F']
    # pred_NIH = pd.read_csv("./NIH/results/bipred.csv")
    # factor_NIH = [gender_NIH, age_decile_NIH]
    # factor_str_NIH = ['Patient Gender', 'Patient Age']

    # ******************************  MIMIC data  ******************************
    # diseases_MIMIC = ['Airspace Opacity', 'Atelectasis', 'Cardiomegaly',
    #                   'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
    #                   'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    #                   'Pneumonia', 'Pneumothorax', 'Support Devices']
    # age_decile_MIMIC = ['60-80', '40-60', '20-40', '80-', '0-20']
    # gender_MIMIC = ['M', 'F']
    # race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
    #         'HISPANIC/LATINO', 'OTHER', 'ASIAN',
    #         'AMERICAN INDIAN/ALASKA NATIVE']
    # insurance = ['Medicare', 'Other', 'Medicaid']
    #
    # pred_MIMIC = pd.read_csv("./results/bipred.csv")
    # factor_MIMIC = [gender_MIMIC, age_decile_MIMIC, race, insurance]
    # factor_str_MIMIC = ['gender', 'age_decile', 'race', 'insurance']

    # ******************************     CXP data   ******************************
    diseases_CXP = ['Lung Opacity', 'Atelectasis', 'Cardiomegaly',
                    'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                    'Pneumothorax', 'Support Devices']
    Age_CXP = ['60-80', '40-60', '20-40', '80-', '0-20']
    gender_CXP = ['M', 'F']
    pred_CXP = pd.read_csv("./results/bipred.csv")
    factor_CXP = [gender_CXP, Age_CXP]
    factor_str_CXP = ['Sex', 'Age']

    for i in range(len(factor_CXP)):
        calculate_cor_CXP_second(pred_CXP, diseases_CXP, factor_CXP[i], factor_str_CXP[i])
    #
    # for i in range(len(factor_NIH)):
    #     calculate_cor_NIH_second(pred_NIH, diseases_NIH, factor_NIH[i], factor_str_NIH[i])

    # for i in range(len(factor_MIMIC)):
    #     calculate_cor_MIMIC_second(pred_MIMIC, diseases_MIMIC, factor_MIMIC[i], factor_str_MIMIC[i])
