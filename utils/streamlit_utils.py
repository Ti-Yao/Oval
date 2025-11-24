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
import re
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from scipy.interpolate import CubicSpline
import pandas as pd
import glob
from scipy.ndimage import zoom
import pydicom
from datetime import datetime

from array_ops import *
from losses import *
from layer_util import *
from unet3plus import *

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

min_timesteps = 10
max_timesteps = 55
min_venc = 20
max_venc = 500

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

def make_video(image, pred_mask, vessel, save_path, alpha = 0.5):
    fig, ax = plt.subplots(1,1, figsize = (5,5))
    frames = []
    for i in range(image.shape[2]):
        p1 = ax.imshow(image[...,i],cmap = 'gray', vmin = np.min(image), vmax = np.max(image))
        text = plt.text(0,-5, f"Frame = {i}")

        ax.axis('off')
        artists = [p1,text]
        artists.append(ax.imshow(pred_mask[...,i],alpha = pred_mask[...,i] * alpha, cmap = colormaps[vessel]))
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
    phase_image = image[...,0] # ************************************************* THIS NEEDS CHANGING IF THE MODEL CHANGES *************************************************
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
    ps = 1 # pixel spacing in mm
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


def calculate_flow(phase_image, mask, rr, vessel):
    flow_curve = calculate_curve(mask, phase_image, vessel)
    flow_curve = interpolate_curve(flow_curve, rr)
    flow = np.mean(flow_curve) * 0.06
    total_volume = np.sum(flow_curve)/1000
    forward_volume = np.sum(flow_curve[flow_curve>0])/1000
    backward_volume = abs(np.sum(flow_curve[flow_curve<0])/1000)
    return flow_curve, flow, total_volume, forward_volume, backward_volume

def convert_time_to_seconds(time_str):
    """
    Converts a time string in the format HHMMSS.FFFFFF into total seconds.
    """
    # Split hours, minutes, seconds, and fractional part
    if '.' in time_str:
        time_part, frac_part = time_str.split('.')
        fractional_seconds = float('0.' + frac_part)
    else:
        time_part = time_str
        fractional_seconds = 0.0

    hours = int(time_part[:2])
    minutes = int(time_part[2:4])
    seconds = int(time_part[4:6])

    total_seconds = hours * 3600 + minutes * 60 + seconds + fractional_seconds
    return total_seconds




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
            dicom_info[dicom_path]['patient'] = str(dcm.PatientName)
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
                try:
                    dicom_info[dicom_path]['creationtime'] =  convert_time_to_seconds(dcm.InstanceCreationTime)
                except:
                    dicom_info[dicom_path]['creationtime'] = convert_time_to_seconds(dcm[0x0008, 0x0033].value)
            except:
                dicom_info[dicom_path]['creationtime'] = 0
                
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
            try:
                dicom_info[dicom_path]['seriesnumber'] = round(dcm.SeriesNumber,3)
            except:
                dicom_info[dicom_path]['seriesnumber'] = series_num
            
        
    
    dicom_info = pd.DataFrame.from_dict(dicom_info, orient = 'index').reset_index().rename(columns={'index': 'dicom'}).sort_values(['triggertime','slicelocation']) # put dicom info for all images into a dataframe
    return dicom_info

class OvalPipeline:
    def __init__(self, data_path, mag_series_uid, phase_series_uid):
        mag_dir = glob.glob(f'{data_path}/**/{mag_series_uid}', recursive=True)[0]
        phase_dir = glob.glob(f'{data_path}/**/{phase_series_uid}', recursive=True)[0]
        mag_dicoms = glob.glob(f'{mag_dir}/**/*.dcm', recursive=True)
        phase_dicoms = glob.glob(f'{phase_dir}/**/*.dcm', recursive=True)
        self.dicom_info = self.get_dicom_info(mag_dicoms, phase_dicoms)
        self.stack_df_list = self.get_stack_df_list(self.dicom_info)
        self.image = self.get_images(self.stack_df_list)


    def get_dicom_info(self, mag_dicoms, phase_dicoms):
        dicoms_in_series = np.unique(mag_dicoms + phase_dicoms)
        dicom_info = read_dicom_header(dicoms_in_series)
        self.manufacturer = dicom_info.iloc[0].manufacturer
        dicom_info = dicom_info.drop(columns = ['manufacturer'])
        dicom_info = dicom_info.dropna(subset=['orientation', 'position'])
        dicom_info['orientation'] = dicom_info['orientation'].apply(tuple)
        dicom_info['position'] = dicom_info['position'].apply(tuple)

        return dicom_info
    
    def match_image_planes(self, dicom_info):
        cine_df_list = []
        grouped = dicom_info.groupby(['orientation', 'position','scale'], dropna=False)

        for _, group_df in grouped:
            enough_frames = len(group_df) > (min_timesteps * 2)
            venc_in_range = ((group_df.venc > min_venc) & (group_df.venc < max_venc)).any()

            if venc_in_range and enough_frames:
                sorted_df = group_df.sort_values(['triggertime', 'creationtime'])
                cine_df_list.append(sorted_df)
        return cine_df_list

    def split_mag_phase_df(self, stack_df):
        mag_df = pd.DataFrame()
        phase_df = pd.DataFrame()

        if 'ge' in self.manufacturer.lower():
            groups = list(stack_df.groupby(stack_df.image.apply(lambda x: x.min() < 0)))
            if len(groups) == 2:
                mag_df, phase_df = [g[1] for g in groups]
            elif len(groups) == 1:
                # Assign group based on the boolean key
                key, df = groups[0]
                if key:  # True group, which one should it be?
                    mag_df = df
                else:
                    phase_df = df
        else:
            groups = list(stack_df.groupby(stack_df['venc'] == 0))
            if len(groups) == 2:
                phase_df, mag_df = [g[1] for g in groups]
            elif len(groups) == 1:
                key, df = groups[0]
                if key:
                    phase_df = df
                else:
                    mag_df = df

        return mag_df, phase_df
        
    def get_multi_series_phase_contrast(self, dicom_info):
        
        stack_df_list = []
        dicom_info = dicom_info.groupby('seriesuid').filter(lambda g: g['position'].nunique() == 1) # clean 4D data
        cine_df_list = self.match_image_planes(dicom_info)
        print(len(cine_df_list))

        for cine_df in cine_df_list:
            mag_df, phase_df = self.split_mag_phase_df(cine_df)
            if mag_df.empty or phase_df.empty:
                self.missing_data_flag = True
                continue


            self.has_creationtime = mag_df['creationtime'].nunique() > 1

            has_creationtime = mag_df['creationtime'].nunique() > 1
            has_seriesnumber = dicom_info.seriesnumber.nunique() > 1

            print(has_creationtime, has_seriesnumber)

            scenario_2 = has_creationtime and not has_seriesnumber
            scenario_3 = not has_creationtime and not has_seriesnumber

            # Match series by closest mean creation time

            if scenario_2 or scenario_3:                
                matches = {}
                mag_times = mag_df.groupby('seriesuid')['creationtime'].mean()
                phase_times = phase_df.groupby('seriesuid')['creationtime'].mean()

                for mag_uid, mag_time in mag_times.items():
                    time_diff = (phase_times - mag_time).abs()
                    min_time_diff = time_diff.min()
                    phase_uids = time_diff.loc[abs(time_diff - min_time_diff) < 0.5].index.tolist()
                    matches[mag_uid] = phase_uids

            else:
                matches = {}
                mag_times = mag_df.groupby('seriesuid')['seriesnumber'].mean()
                phase_times = phase_df.groupby('seriesuid')['seriesnumber'].mean()
                for mag_uid, mag_time in mag_times.items():
                    time_diff = (phase_times - mag_time).abs()
                    min_time_diff = time_diff.min()
                    phase_uids = time_diff.loc[abs(time_diff - min_time_diff) <= 2].index.tolist()
                    matches[mag_uid] = phase_uids


            for mag_series_uid, mag_series_df in mag_df.groupby('seriesuid'):
                phase_series_uids = matches[mag_series_uid]
                phase_series_df = phase_df.loc[phase_df['seriesuid'].isin(phase_series_uids)]
                print(len(phase_series_df), len(mag_series_df))

                if len(mag_series_df) == len(phase_series_df):
                    self.complex_difference = False
                    stack_df = pd.concat([mag_series_df, phase_series_df])

                elif len(mag_series_df) == len(phase_series_df) * 2:
                    self.complex_difference = True
                    print('complex difference')

                    max_vals = np.max(np.stack(mag_series_df['image'].values, -1))
                    mag_series_df['norm_sum'] = mag_series_df['image'].apply(
                        lambda x: np.sum(x / max_vals)
                    )

                    chosen_mag_df = mag_series_df.loc[
                        mag_series_df.groupby('triggertime')['norm_sum'].idxmax()
                    ].drop(columns='norm_sum').reset_index(drop=True)

                    stack_df = pd.concat([chosen_mag_df, phase_series_df])

                elif len(mag_series_df) * 2 == len(phase_series_df):
                    self.complex_difference = True
                    print('complex difference')
                    phase_series_df['min_val'] = phase_series_df['image'].apply(
                        lambda x: np.min(x)
                    )

                    chosen_phase_series_df = (
                        phase_series_df.loc[
                            phase_series_df.groupby('triggertime')['min_val'].idxmin()
                        ]
                        .drop(columns='min_val')
                        .reset_index(drop=True)
                    )
                    stack_df = pd.concat([mag_series_df, chosen_phase_series_df])
                    
                else:
                    self.missing_data_flag = True
                    continue

                if len(stack_df) >= 2 * min_timesteps and len(stack_df) <= 2 * max_timesteps:
                    stack_df_list.append(stack_df)

        return stack_df_list

    def get_stack_df_list(self, dicom_info):
        stack_df_list = self.get_multi_series_phase_contrast(dicom_info)
        for stack_df in stack_df_list[:]:
            if (stack_df['rr'] == 0).all():
                max_tt = stack_df.triggertime.max()
                diff_tt = stack_df['triggertime'].diff().dropna()
                diff_tt = diff_tt.loc[diff_tt > 0]
                diff_tt = diff_tt.mode()[0]
                rr = max_tt + diff_tt
                stack_df['rr'] = rr
            
            elif (stack_df['rr'] == 0).any():
                rrs = stack_df['rr']
                rrs = rrs.loc[rrs > 0]
                rr = rrs.median()
                stack_df['rr'] = rr

        if len(stack_df_list) == 0:
            raise ValueError('No Phase-Contrasts')
        
        return stack_df_list


        
    def get_images(self, stack_df_list):
        print(len(stack_df_list))
        if len(stack_df_list) != 1:
            raise ValueError('Multiple Phase-Contrast Series Found. Please select only one series.')
        
        stack_df = stack_df_list[0] # there should only be one series

        if 'ge' in self.manufacturer.lower(): # split phase-contrast into mag and phase
            mag_df, phase_df = [x for _ , x in stack_df.groupby(stack_df.image.apply(lambda x: x.min()< 0))]
        else:
            phase_df, mag_df = [x for _ , x in stack_df.groupby(stack_df['venc'] == 0)]

        mag_df = mag_df.drop_duplicates(subset = ['uid'])
        mag_tt = mag_df.triggertime.unique()
        mag_df = mag_df.loc[mag_df['seriesuid'] == mag_df['seriesuid'].iloc[0]]
        phase_df = phase_df.drop_duplicates(subset = ['uid'])
        phase_df = phase_df[phase_df['triggertime'].isin(mag_tt)]
        phase_df = phase_df.loc[phase_df['venc'] == np.max(phase_df['venc'])]

        phase_image = np.stack(phase_df['image'], -1)
        mag_image = np.stack(mag_df['image'], -1)

        if 'ge' in self.manufacturer.lower(): # GE needs extra processing for velocity
            vas_flag = phase_df.iloc[0]['vas_flag']
            if vas_flag != 0:
                scale = phase_df.iloc[0]['scale']
                velocity = np.divide(phase_image, mag_image, out=np.zeros_like(phase_image, dtype=float), where=mag_image != 0) * scale
                phase_image = velocity 
                
        ps = float(phase_df.iloc[0].pixelspacing[0])
        frames = mag_image.shape[-1]
        ratio = 32/frames
        mag_image = zoom(mag_image, [ps, ps, ratio], order = 1) # resize magnitude to 1x1x1mm
        phase_image = zoom(phase_image, [ps, ps, ratio], order = 1) # resize phase to 1x1x1mm

        max_size = 256

        mag_image = tf.image.resize_with_crop_or_pad(mag_image, max_size, max_size) # crop or pad images to 256x256
        phase_image = tf.image.resize_with_crop_or_pad(phase_image, max_size, max_size)  # crop or pad images to 256x256

        image = np.stack([mag_image, phase_image], -1) # combine magnitude and phase together
        self.venc = phase_df.venc.max()
        self.rr =  phase_df.rr.max()
        description = mag_df.seriesdescription.iloc[0]
        self.description = description if description else "[empty]"
        self.patient =  mag_df.patient.iloc[0]
        studydate =  mag_df.studydate.iloc[0]
        self.studydate = datetime.strptime(studydate, "%Y%m%d").strftime("%m-%d-%Y")
        return image


def combine_gif_png(gif_path, png_path, output_path):
    """
    Combine the segmentation GIF and the flow curve PNG side by side,
    resizing the PNG relative to the GIF height by a scale factor.
    """
    scale_factor=1
    gif = Image.open(gif_path)
    png = Image.open(png_path).convert("RGBA")

    gif_height = gif.height
    target_height = int(gif_height * scale_factor)

    # Resize PNG with aspect ratio preserved
    png_width = int(png.width * (target_height / png.height))
    png_resized = png.resize((png_width, target_height), Image.LANCZOS)

    frames = []
    try:
        while True:
            gif_frame = gif.copy().convert("RGBA")
            combined = Image.new("RGBA", (gif.width + png_resized.width, gif_height))

            # Paste GIF and vertically center PNG
            combined.paste(gif_frame, (0, 0))
            y_offset = (gif_height - target_height) // 2
            combined.paste(png_resized, (gif.width, y_offset))
            frames.append(combined)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)