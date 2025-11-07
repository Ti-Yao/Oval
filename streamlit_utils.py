import os
import json
import base64
import tensorflow as tf
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import skimage
import nibabel as nib
from PIL import Image, ImageDraw

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from losses import *
from layer_util import *
from unet3plus import *
from scipy.interpolate import CubicSpline
import pandas as pd
import glob
from scipy.ndimage import zoom
import pydicom
from datetime import datetime

vessels_dict = {'lpa':1,'rpa':2,'ao':3,'svc':4,'ivc':5}
skip = 5
red = '#FF4E32'
blue = '#57C7FF'
green = '#A2F55A'
purple = '#CD68FF'
orange = '#FF771C'
colorlist = {
    'lpa': red,
    'rpa': blue,
    'ao': green,
    'svc': purple,
    'ivc': orange
}

colormaps = {
    'lpa': ListedColormap([red]),
    'rpa': ListedColormap([blue]),
    'ao': ListedColormap([green]),
    'svc': ListedColormap([purple]),
    'ivc': ListedColormap([orange])
}
colormaps = {k:colormaps[k] for k in vessels_dict.keys()}

image_size = 96
frames = 32



def normalise(image):
    if np.max(image) != 0:
        norm = (image - np.min(image)) / (np.max(image)-np.min(image))
        return norm
    else:
        return image
    
def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def phase2angle(value, venc, to_range=(-np.pi, np.pi)):
    from_min = -venc
    from_max = venc
    to_min, to_max = to_range
    mapped_value = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value

def create_complex_image(magnitude, phase): # magnitude is a real number tensor; phase is a tensor of radiant angles
#     if np.max(phase) > (np.pi+1e-2) or np.min(phase) < -(np.pi+1e-2):
#         print('Not right about phase')
    # Calculate real and imaginary parts
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    
    # Create complex image
    complex_image = np.stack((real_part, imag_part), axis=-1)
    
    return complex_image

def make_video(image, pred_mask, vessel, save_path):
    fig, ax = plt.subplots(1,1, figsize = (5,5))
    frames = []
    for i in range(image.shape[2]):
        p1 = ax.imshow(image[...,i],cmap = 'gray', vmin = np.min(image), vmax = np.max(image))
        text = plt.text(0,-5, f"Frame = {i}")

        ax.axis('off')
        artists = [p1,text]
        artists.append(ax.imshow(pred_mask[...,i],alpha = pred_mask[...,i] * 0.5, cmap = colormaps[vessel]))
        frames.append(artists)
    ani = animation.ArtistAnimation(fig, frames)
    plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    ani.save(f'{save_path}', fps=image.shape[2])
    plt.close()
        

def get_crop_coords(coords):
    x, y = coords['x'], coords['y']
    x_min, x_max = x - image_size//2, x + image_size//2
    y_min, y_max = y - image_size//2, y + image_size//2
    return x_min, x_max, y_min, y_max

def segment_image(image, venc, model, x_min, x_max, y_min, y_max):
    mag_image = image[...,0]
    phase_image = image[...,0]
    mag_image[mag_image<1e-10] = 0                
    max_val = np.max(phase_image)
    angles = phase2angle(phase_image, venc)
    mag_image = (mag_image - np.min(mag_image))/(np.max(mag_image) - np.min(mag_image))
    mag_image[mag_image>=1] = 1

    phase_image = phase_image.astype('float32')/max_val
    mag_image = mag_image[y_min:y_max, x_min:x_max,:]
    phase_image = phase_image[y_min:y_max, x_min:x_max,:]

    angles = phase2angle(phase_image, venc)

    mag_image = skimage.exposure.equalize_adapthist(mag_image)
    complex_image = create_complex_image(mag_image, angles)
    real_image, imaginary_image = complex_image[...,0],complex_image[...,1]

    mag_image = normalise(mag_image)        
    imaginary_image = normalise(imaginary_image)        
    phase_image = normalise(phase_image)        


    X = np.stack([mag_image, imaginary_image], -1)
    pred_mask = model.predict(X[np.newaxis])[-1][0][...,0]
    pred_mask[pred_mask<0.5] = 0
    pred_mask[pred_mask>=0.5] = 1

    # resize 
    full_pred_mask = np.zeros_like(image[...,0], dtype=np.uint8)
    full_pred_mask[y_min:y_max, x_min:x_max] = pred_mask.astype(np.uint8)

    return mag_image, pred_mask, full_pred_mask

def calculate_curve(mask, phase_image, vessel):
    ps = 1 if mask.shape[0] == 256 else 2
    pixel_area = ps **2 / 100  # convert mm2 to cm2
    phase = mask * phase_image
    v_curve = np.sum(np.sum(phase*pixel_area ,0),0) # cm3/s
    if np.mean(v_curve) < 0:
        v_curve = -v_curve
    return v_curve


def interpolate_curve(curve, phase_vessel_rr):
    x_original = np.linspace(0, round(phase_vessel_rr), len(curve))
    y_original = curve
    cs = CubicSpline(x_original, y_original)
    x_new = np.arange(0, round(phase_vessel_rr))
    return cs(x_new)

def get_crop_coords(coords):
    x, y = coords['x'], coords['y']
    x_min, x_max = x - image_size//2, x + image_size//2
    y_min, y_max = y - image_size//2, y + image_size//2
    return x_min, x_max, y_min, y_max

def display_gif(file_path, width):
    """Display a GIF in Streamlit from a file path."""
    with open(file_path, "rb") as f:
        contents = f.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" width="{width}">', unsafe_allow_html=True)


def get_ellipse_coords(point, radius=5):
    x, y = point
    return (x - radius, y - radius, x + radius, y + radius)



def read_dicom_header(dicoms_in_series):
    '''
    read the information we want from the header and assert that the series has to have pixelarray data
    '''
    dicom_info = {}
    for dicom_path in dicoms_in_series: # go through dicom in each series
        dcm = pydicom.dcmread(dicom_path, force=True) # read dicom

        try: # if dicom doesn't have an associate pixel array (image), ignore dicom
            image = dcm.pixel_array  
            image_exists = True
            if image.ndim == 3: # ignore dicom if 3d
                image_exists = False
            try:
                if dcm.MRAcquisitionType == '3D': # ignore dicom if 3d
                    image_exists = False
                    break
            except:
                pass

        except Exception as e:
            print('error reading image', e)
            image_exists = False
            break

        if image_exists: # if image exists and is not 3d read all other information
            dicom_info[dicom_path] = {}
            image = dcm.pixel_array.astype('float32')
            try:
                intercept = dcm.RescaleIntercept
                slope = dcm.RescaleSlope
            except:
                try:
                    intercept = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept 
                    slope = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueSlope 
                except:
                    intercept = 1
                    slope = 1
            image = image * slope + intercept # true pixel values from dicom
            manufacturer = dcm.Manufacturer.lower()
            
            # initialise some variables
            venc = 0
            scale = 1
            vas_flag = 0
            
            if 'siemens' in manufacturer:
                try:
                    venc = str(dcm[0x0051, 0x1014]._value)
                    numbers = re.findall(r'\d+', venc)
                    venc = float(max(list(map(int, numbers))))
                except:
                    try:
                        venc = str(dcm[0x0018, 0x0024]._value)
                        venc = float(re.search(r'v(\d+)in', venc).group(1))
                    except:
                        venc = 0
                image = image.astype('float32')
                if venc > 0:
                    image = (image * venc)/4096
                    
            if 'ge' in manufacturer:
                try:
                    venc = dcm[0x0019, 0x10cc].value/10 
                    vas_flag = dcm[0x0043, 0x1032]._value
                    venc_scale = float(dcm[0x0019, 0x10E2]._value)
                    
                    vas_flag = 2 & vas_flag
                    
                    if vas_flag != 0:
                        scale = venc/(venc_scale * np.pi)
                    if vas_flag == 0 and venc >0:
                        image = image/10

                except:
                    venc = 0
                    
            if 'philips' in manufacturer:
                try:
                    venc = abs(list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept)
                except:
                    try:
                        venc = abs(dcm.RescaleIntercept)
                    except:
                        venc = 0

            dicom_info[dicom_path]['venc'] = float(venc)
            dicom_info[dicom_path]['vas_flag'] = vas_flag
            dicom_info[dicom_path]['scale'] = scale
            dicom_info[dicom_path]['image'] = image
            dicom_info[dicom_path]['uid'] =  '.'.join(dcm.SOPInstanceUID.split('.')[-2:])
            dicom_info[dicom_path]['seriesuid'] =  dcm.SeriesInstanceUID
            dicom_info[dicom_path]['manufacturer'] = manufacturer.lower()
            dicom_info[dicom_path]['patient'] = dcm.PatientName
            dicom_info[dicom_path]['studydate'] = dcm.StudyDate
            
            
            rr_ni, rr_hr = 0, 0
            try:
                rr_ni = round(dcm.NominalInterval,3)
            except Exception as e:
                rr_ni = 0
            try:
                rr_hr = round(60000/dcm.HeartRate,3)
            except Exception as e:
                rr_hr = 0
            rr = np.max([rr_ni, rr_hr])
            rr = rr if (rr > 100) and (rr < 3000) else 0
            dicom_info[dicom_path]['rr'] = rr

            try:
                dicom_info[dicom_path]['seriesdescription'] = dcm.SeriesDescription.lower()
            except:
                dicom_info[dicom_path]['seriesdescription'] = ''
            try:
                dicom_info[dicom_path]['pixelspacing'] = dcm.PixelSpacing
            except:
                dicom_info[dicom_path]['pixelspacing'] = ''
            try:
                dicom_info[dicom_path]['triggertime'] = round(dcm.TriggerTime)
            except:
                dicom_info[dicom_path]['triggertime'] = np.nan
            try:
                dicom_info[dicom_path]['orientation'] = [round(val,3) for val in dcm.ImageOrientationPatient]
            except:
                dicom_info[dicom_path]['orientation'] = np.nan
            try:
                dicom_info[dicom_path]['position'] = [round(val,3) for val in dcm.ImagePositionPatient]
            except:
                dicom_info[dicom_path]['position'] = np.nan
            try:
                dicom_info[dicom_path]['slicelocation'] = round(dcm.SliceLocation,3)
            except:
                dicom_info[dicom_path]['slicelocation'] = np.nan
    dicom_info = pd.DataFrame.from_dict(dicom_info, orient = 'index').reset_index().rename(columns={'index': 'dicom'}).sort_values(['triggertime','slicelocation']) # put dicom info for all images into a dataframe
    return dicom_info, manufacturer



def get_image(data_path, mag_series_uid, phase_series_uid):
    mag_dir = glob.glob(f'{data_path}/**/{mag_series_uid}', recursive=True)[0]
    phase_dir = glob.glob(f'{data_path}/**/{phase_series_uid}', recursive=True)[0]
    mag_dicoms = glob.glob(f'{mag_dir}/**/*.dcm', recursive=True)
    phase_dicoms = glob.glob(f'{phase_dir}/**/*.dcm', recursive=True)
    dicoms_in_series = np.unique(mag_dicoms + phase_dicoms)

    dicom_info, manufacturer = read_dicom_header(dicoms_in_series)

    if 'ge' in manufacturer.lower(): # split phase-contrast into mag and phase
        mag_df, phase_df = [x for _ , x in dicom_info.groupby(dicom_info.image.apply(lambda x: x.min()< 0))]
    else:
        phase_df, mag_df = [x for _ , x in dicom_info.groupby(dicom_info['venc'] == 0)]

    mag_df = mag_df.drop_duplicates(subset = ['uid'])
    mag_tt = mag_df.triggertime.unique()
    mag_df = mag_df.loc[mag_df['seriesuid'] == mag_df['seriesuid'].iloc[0]]
    phase_df = phase_df.drop_duplicates(subset = ['uid'])
    phase_df = phase_df[phase_df['triggertime'].isin(mag_tt)]
    phase_df = phase_df.loc[phase_df['venc'] == np.max(phase_df['venc'])]

    phase_image = np.stack(phase_df['image'], -1)
    mag_image = np.stack(mag_df['image'], -1)

    if 'ge' in manufacturer.lower(): # GE needs extra processing for velocity
        vas_flag = phase_df.iloc[0]['vas_flag']
        if vas_flag != 0:
            scale = phase_df.iloc[0]['scale']
            velocity = np.divide(phase_image, mag_image, out=np.zeros_like(phase_image, dtype=float), where=mag_image != 0) * scale
            phase_image = velocity 
            print('magnitude-weighted')
            
    ps = float(phase_df.iloc[0].pixelspacing[0])
    frames = mag_image.shape[-1]
    ratio = 32/frames
    mag_image = zoom(mag_image, [ps, ps, ratio], order = 1) # resize magnitude to 1x1x1mm
    phase_image = zoom(phase_image, [ps, ps, ratio], order = 1) # resize phase to 1x1x1mm

    max_size = 256

    mag_image = tf.image.resize_with_crop_or_pad(mag_image, max_size, max_size) # crop or pad images to 256x256
    phase_image = tf.image.resize_with_crop_or_pad(phase_image, max_size, max_size)  # crop or pad images to 256x256

    image = np.stack([mag_image, phase_image], -1) # combine magnitude and phase together
    venc = phase_df.venc.max()
    rr =  phase_df.rr.max()
    description =  mag_df.seriesdescription.iloc[0]
    patient =  mag_df.patient.iloc[0]
    study_date =  mag_df.studydate.iloc[0]
    study_date = datetime.strptime(study_date, "%Y%m%d").strftime("%m-%d-%Y")

    return image, venc, rr, description, patient, study_date
