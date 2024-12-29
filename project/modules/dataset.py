import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

class CD_TrippletsCreator(Dataset):
    
    def __init__(self, dataframe, transform=None, *, data_augmentation_tranforms=[]):
        
        self.transform = transform
        self.data_augmentation_transforms = data_augmentation_tranforms

        self.dataframe = pd.DataFrame(dataframe, columns=[*dataframe.columns, 'augm'])
        self.dataframe.augm = None

        for idx in range(len(data_augmentation_tranforms)):
            tmp = pd.DataFrame(dataframe, columns=[*dataframe.columns, 'augm'])
            tmp.augm = idx
            self.dataframe = pd.concat([self.dataframe, tmp], ignore_index=True, sort=False)

    
    def __len__(self):
        return self.dataframe.count(axis=1).size
    
    
    def __getitem__(self, idx):

        # anchor
        anchor = self.dataframe.iloc[idx]
        anchor_path, anchor_setid, anchor_augm = anchor.path, anchor.setid, anchor.augm

        # positive
        posit = self.dataframe[(self.dataframe.setid == anchor_setid) & (self.dataframe.index != anchor.name)].sample(1).iloc[0]
        posit_path, posit_augm = posit.path, posit.augm

        # negative
        negat = self.dataframe[self.dataframe.setid != anchor_setid].sample(1).iloc[0]
        negat_path, negat_setid, negat_augm = negat.path, negat.setid, negat.augm


        imgs = [read_image(anchor_path), read_image(posit_path), read_image(negat_path)]
        imgs_augm = (anchor_augm, posit_augm, negat_augm)

        # data augmentation
        for i, augm in enumerate(imgs_augm):
            if augm is not None:
                imgs[i] = self.data_augmentation_transforms[augm](imgs[i])

        # normalization
        if self.transform: 
            for i, img in enumerate(imgs): imgs[i] = self.transform(img)

        return *imgs, anchor_setid, negat_setid