import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import cv2

import torch
import glob
import os
import re

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

import torch.nn as nn

import random

#import efficientnet_pytorch

#from sklearn.model_selection import StratifiedKFold
#from torchvision import transforms
import monai

def predict(patient_id):
    NUM_IMAGES_3D = 64
    TRAINING_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    IMAGE_SIZE = 224
    N_EPOCHS = 40
    do_valid = True
    n_workers = 1
    patient_id = 00000

# mkdir indices

# df = pd.read_csv("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv")
# df.head()

    def extract_cropped_image_size(path):
        """
        reading dicom files and returning the resolution after cropping the files using `crop_img` function 
        resolution : number of pixels in cropped dicom file
        """
        
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        cropped_data = crop_img(data)
        resolution = cropped_data.shape[0]*cropped_data.shape[1]  
        return resolution


    def crop_img(img):
        
        """
        removing zero valued pixels in dicom slice , if the dicom file is all zeros the fucntion returns an empty list
        
        
        """
        
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        c1, c2 = False, False
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]        # np.where(rows) : gettin indices of True values (not zero pixels in dicom)
        except:                                            # np.where(rows)[0][0,-1] getting the first and the last indices of the non zero pixeles in dicom file  (rmin , rmax)
            rmin, rmax = 0, img.shape[0]                   # remove all zeros slices           
            c1 = True

        try:
            cmin, cmax = np.where(cols)[0][[0, -1]]
        except:
            cmin, cmax = 0, img.shape[1]
            c2 = True
        bb = (rmin, rmax, cmin, cmax)
        
        if c1 and c2:
            return img[0:0, 0:0]                           # remove all zeros slices
        else:
            return img[bb[0] : bb[1], bb[2] : bb[3]]


    def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        if rotate > 0:
            rot_choices = [
                0,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
                cv2.ROTATE_180,
            ]
            data = cv2.rotate(data, rot_choices[rotate])

        data = cv2.resize(data, (img_size, img_size))
        data = data - np.min(data)
        if np.min(data) < np.max(data):
            data = data / np.max(data)
        return data


    def crop_img(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        c1, c2 = False, False
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
        except:
            rmin, rmax = 0, img.shape[0]
            c1 = True

        try:
            cmin, cmax = np.where(cols)[0][[0, -1]]
        except:
            cmin, cmax = 0, img.shape[1]
            c2 = True
        bb = (rmin, rmax, cmin, cmax)
        
        if c1 and c2:
            return img[0:0, 0:0]
        else:
            return img[bb[0] : bb[1], bb[2] : bb[3]]



    def extract_cropped_image_size(path):
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        cropped_data = crop_img(data)
        resolution = cropped_data.shape[0]*cropped_data.shape[1]  
        return resolution

    class BrainRSNADataset(Dataset):
        def __init__(
            self, data, transform=None, target="MGMT_value", mri_type="T2w", is_train=True, ds_type="forgot", do_load=True
        ):
            self.target = target
            self.data = data
            self.type = mri_type

            self.transform = transform
            self.is_train = is_train
            self.folder = "train" if self.is_train else "test"
            self.do_load = do_load
            self.ds_type = ds_type
            self.img_indexes = self._prepare_biggest_images()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.loc[index]
            case_id = int(row.BraTS21ID)
            target = int(row[self.target])
            _3d_images = self.load_dicom_images_3d(case_id)
            _3d_images = torch.tensor(_3d_images).float()
            if self.is_train:
                return {"image": _3d_images, "target": target, "case_id": case_id}
            else:
                return {"image": _3d_images, "case_id": case_id}
        
    
        
        def _prepare_biggest_images(self):
            """
            getting the biggest dicom file from patient scans after cropping zero valued pixels
            
            """
            
            
            big_image_indexes = {}
            if (f"big_image_indexes_{patient_id}.pkl" in os.listdir("indices/"))\
                and (self.do_load) :
                print("Loading the best images indexes for all the cases...")
                big_image_indexes = joblib.load(f"indices/big_image_indexes_{patient_id}.pkl")
                return big_image_indexes
            else:
                
                print("Caulculating the best scans for every case...")
                for row in tqdm(self.data.iterrows(), total=len(self.data)):
                    case_id = str(int(row[1].BraTS21ID)).zfill(5)
                    path = f"uploads/{patient_id}/*.dcm"
                    files = sorted(
                        glob.glob(path),
                        key=lambda var: [
                            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                        ],
                    )
                    resolutions = [extract_cropped_image_size(f) for f in files]
                    middle = np.array(resolutions).argmax()
                    big_image_indexes[case_id] = middle

                joblib.dump(big_image_indexes, f"indices/big_image_indexes_{patient_id}.pkl")
                return big_image_indexes


        
        def load_dicom_images_3d(
            self,
            case_id,
            num_imgs=NUM_IMAGES_3D,
            img_size=IMAGE_SIZE,
            rotate=0,
        ):
            case_id = str(case_id).zfill(5)

            path = f"uploads/{patient_id}/*.dcm"
            files = sorted(
                glob.glob(path),
                key=lambda var: [
                    int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )

        
            middle = self.img_indexes[case_id]   # largest resolution index of cropped dicom files  (largest cropped dicom)

            middle = len(files) // 2
            num_imgs2 = num_imgs // 2
            p1 = max(0, middle - num_imgs2)    # if the largest resultion dicom index less than the half of image depth start from 0
            p2 = min(len(files), middle + num_imgs2)  # either you take all files of only half the depth after the largest
            image_stack = [load_dicom_image(f, rotate=rotate, voi_lut=True) for f in files[p1:p2]]  #stacking images after one another
            
            img3d = np.stack(image_stack).T
            if img3d.shape[-1] < num_imgs:   # in case all the dicom files are less than the preset `num_imgs` 
                n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
                img3d = np.concatenate((img3d, n_zero), axis=-1)

            return np.expand_dims(img3d, 0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=2)

    model.class_layers.out = nn.Linear(1024 , 1)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("./0.6303030303030303_acc_model_resnet50 DataParallel_best_metric_model.pth"))




    df_uni = pd.DataFrame({"BraTS21ID":[patient_id], "MGMT_value":[0.5]})


    df_uni_ds = BrainRSNADataset(data=df_uni, mri_type="FLAIR",ds_type="val" , is_train= False)

    df_uni_dl =  torch.utils.data.DataLoader(
        df_uni_ds,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    
    )



    with torch.no_grad():
            val_loss = 0.0
            preds = []
            true_labels = []
            case_ids = []
            epoch_iterator_val = tqdm(df_uni_dl)
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                images = batch["image"].to(device)#, batch["target"].to(device)

                outputs = model(images)
            # targets = targets  # .view(-1, 1)
            # loss = criterion(outputs.squeeze(1), targets.float())
            # val_loss += loss.item()
            # epoch_iterator_val.set_postfix(
            #    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
            # )
                preds.append(outputs.sigmoid().detach().cpu().numpy())
            #  true_labels.append(targets.cpu().numpy())
                case_ids.append(batch["case_id"])
            preds = np.vstack(preds).T[0].tolist()
            #true_labels = np.hstack(true_labels).tolist()
            case_ids = np.hstack(case_ids).tolist()
            for i in range(len(preds)):
                if preds[i]< 0.5 :
                    preds[i]= 0 
                else:
                    preds[i]=1



    return preds