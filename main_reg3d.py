# -*- coding: utf-8 -*-
"""

This is the main script that invokes the registration code file
"""
from deepReg import deepRigid3d as dr3
from nilearn.image import load_img, resample_img
from nilearn.plotting import plot_anat
import numpy as np 

if __name__ == '__main__':
    img = load_img('/deneb_disk/studyforrest_bfp/sub-01/\
anat/sub-01_T1w.bse.nii.gz')
    plot_anat(img)
    aff = np.dot(np.diag([img.shape[0]/64.0, img.shape[1]/64.0,
                          img.shape[2]/64.0, 1]), np.eye(4))
    img2 = resample_img(img, target_affine=aff, target_shape=(64, 64, 64))
    plot_anat(img2)

    dr3.train_model(img2.get_data())

#    dr3.test_model()
