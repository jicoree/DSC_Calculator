from http.client import REQUESTED_RANGE_NOT_SATISFIABLE

import cv2, os, argparse
from cv2 import CV_32F
from cv2 import Mat
import pydicom
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


root = Tk()
root.title("DSC Calculator")
root.resizable(False, False)

def DCMUploadBtn():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("DCM 파일", "*.DCM"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    for file in files:
        dcm_list_file = files
        return

def annotation_Upload():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    for file in files:
        annotation_list_file = files
        return

def prediction_Upload():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    for file in files:
        prediction_list_file = files
        return
        
def destination_path():
    folder_selected = filedialog.askdirectory()
    if folder_selected is None: # 사용자가 취소 누를 시
        return
    txt_dest_path.delete(0, END)
    txt_dest_path.insert(0, folder_selected)
    
def _dice(im1, im2):
    
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())    


def viz_all(input_img, gt_mask, pred_mask,
            save_to='./output', fn='sample',
            img_cm='gray', gt_cm='Greens', pred_cm='Reds',
            use_contour=False,
            use_grid=True, grid_alpha=0.5
           ):
    '''
    "input_img" -> dicom 이미지의 np.array [1024x1024]
    "gt_mask" -> 정답 마스크의 np.array (min=0, max=1) [1024x1024]
    "pred_mask" -> 모델이 예측한 마스크의 np.array (min=0, max=1) [1024x1024]
    "save_to" -> 모든 이미지의 오버레이와 dice 스코어가 저장되는 폴더
    "fn" -> dicom 이미지 파일의 고유 명칭
    "img_cm" -> dicom 이미지 color map
    "gt_cm" -> 정답 마스크 color map
    "pred_cm" -> 예측 마스크의 color map
    "use_contour" -> 마스크 테두리 표시 여부
    "use_grid" -> 이미지의 그리드 표시 여부
    "grid_alpha" -> 이미지 그리드 transparency
    '''
    
    
    dice_score = _dice(gt_mask, pred_mask)
    dice_score = round(dice_score, 4)
    print("Dice score: {}".format(dice_score))
    
    os.makedirs(save_to, exist_ok=True)
    
    plt.figure(figsize=(30, 18))

    plt.subplot(2,3,1)
    plt.imshow(input_img, cmap=img_cm)
    plt.title("Input", size=22)

    plt.subplot(2,3,2)
    plt.imshow(gt_mask, cmap=gt_cm)
    plt.title("Mask-Ground Truth", size=22)

    plt.subplot(2,3,3)
    plt.imshow(pred_mask, cmap=pred_cm)
    plt.title("Mask-Prediction", size=22)

    plt.subplot(2,3,4)
    plt.imshow(input_img, cmap=img_cm)
    plt.imshow(gt_mask==1, cmap=gt_cm, alpha=0.17)
    plt.imshow(pred_mask==1, cmap=pred_cm, alpha=0.17)
    plt.title("OL-All", size=22)

    plt.subplot(2,3,5)
    plt.imshow(input_img, cmap=img_cm)
    plt.imshow(gt_mask, cmap=gt_cm, alpha=0.2)
    plt.title("OL-Ground Truth", size=22)

    plt.subplot(2,3,6)
    plt.imshow(input_img, cmap=img_cm)
    plt.imshow(pred_mask, cmap=pred_cm, alpha=0.2)
    plt.title("OL-Prediction", size=22)

    
    if use_grid == True:
        plt.subplot(2,3,1)
        plt.grid(alpha=grid_alpha)
        
        plt.subplot(2,3,2)
        plt.grid(alpha=grid_alpha)
        
        plt.subplot(2,3,3)
        plt.grid(alpha=grid_alpha)
        
        plt.subplot(2,3,4)
        plt.grid(alpha=grid_alpha)
        
        plt.subplot(2,3,5)
        plt.grid(alpha=grid_alpha)
        
        plt.subplot(2,3,6)
        plt.grid(alpha=grid_alpha)
        
    if use_contour == True:
        plt.subplot(2,3,2)
        plt.contour(gt_mask, cmap=gt_cm)
        
        plt.subplot(2,3,3)
        plt.contour(pred_mask, cmap=pred_cm)
        
        plt.subplot(2,3,4)
        plt.contour(gt_mask, cmap=gt_cm)
        plt.contour(pred_mask, cmap=pred_cm)
        
        plt.subplot(2,3,5)
        plt.contour(gt_mask, cmap=gt_cm)
        
        plt.subplot(2,3,6)
        plt.contour(pred_mask, cmap=pred_cm)

    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_to, str(dice_score)+'_'+fn+'.png'))
    
    

def start():
    if annotation_list_file == 0:
        msgbox.showwarning("경고", "마스크 파일을 추가하세요.")
        return
    if prediction_list_file == 0:
        msgbox.showwarning("경고", "마스크 파일을 추가하세요.")
        return
        
    image_path = dcm_list_file
    gt_mask_path = annotation_list_file
    pred_mask_path = prediction_list_file
    
    # file_name = os.path.splitext(image_path)[0].split("/")[-1] 
    file_name = image_path.split('/')[-1].split('.')[0]
    
    '''
    동시에 여러장을 테스트 할 경우는 별도 구현 필요.
    이미지와 마스크를 np.array로 변환하는 함수는 구현하지 않음.
    '''
    
    '''
    dicom 이미지
    img -> np.array, dtype=uint8
    '''
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (1024, 1024))
    
    '''
    정답 마스크
    mask -> np.array, dtype=uint8 (min=0, max=1)
    '''
    mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (1024, 1024))
    mask = mask // mask.max()
    
    '''
    예측 마스크
    pmask -> np.array, dtype=uint8 (min=0, max=1)
    '''
    pmask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    pmask = cv2.resize(pmask, (1024, 1024))
    pmask = pmask // pmask.max()
    
    
    '''
    plot 순서 (저장 파일명: dicescore + dicom 고유명칭 / dicom 고유명칭 별도 입력 필요.)
    '''
    viz_all(img, mask, pmask,
            save_to=opt.save_to, 
            fn=file_name,
            img_cm='gray', 
            gt_cm='Greens', 
            pred_cm='Reds',
            use_contour=False,
            use_grid=True, 
            grid_alpha=0.5)
    
    pass    

def before_image():
    global i
    i = i - 1
    try:
        pass
    except:
        i = 0
        before_image()
    
def after_image():
    global i
    i = i + 1
    try:
        pass
    except:
        i = -1
        after_image()

i = 0
dcm_list_file = list()
annotation_list_file = list()
prediction_list_file = list()

# File upload Frame

File_upload_frame = LabelFrame(root)
File_upload_frame.pack()

# Before Image frame

Before_Image_Button = Button(File_upload_frame, width = 5, text='<', command = before_image)
Before_Image_Button.pack(side = "left", fill = "y")


# DCM file 업로드 frame
DCM_frame = LabelFrame(File_upload_frame, text = "DICOM")
DCM_frame.pack(side = "left", padx = 5, pady = 5)

# DCMImage_frame = Frame(DCM_frame, width = 400, height = 400, bg = "white")
# DCMImage_frame.pack(padx = 5, pady = 5)
# DCMImage_Label = Label(DCMImage_frame, image = cv2.imshow(dcm_list_file[i], Mat))

DCMImage = Button(DCM_frame, width = 20, height = 1, text = "Upload", command = DCMUploadBtn)
DCMImage.pack(padx = 5, pady = 5)



# annotation 업로드 frame
annotation_frame = LabelFrame(File_upload_frame, text = "Annotation")
annotation_frame.pack(side = "left", padx = 5, pady = 5)

# annotationImage_frame = Frame(annotation_frame, width = 400, height = 400, bg = "white")
# annotationImage_frame.pack(padx = 5, pady = 5)
# annotationImage_Label = Label(annotationImage_frame, image = cv2.imshow(annotation_list_file[i], Mat))

annotationImage = Button(annotation_frame, width = 20, height = 1, text = "Upload", command = annotation_Upload)
annotationImage.pack(padx = 5, pady = 5)


# prediction 업로드 frame
prediction_frame = LabelFrame(File_upload_frame, text = "Prediction")
prediction_frame.pack(side = "left", padx = 5, pady = 5)

# predictionImage_frame = Frame(prediction_frame, width = 400, height = 400, bg = "white")
# predictionImage_frame.pack(padx = 5, pady = 5)
# predictionImage_Label = Label(annotationImage_frame, image = cv2.imshow(prediction_list_file[i], Mat))

predictionImage = Button(prediction_frame, width = 20, height = 1, text = "Upload", command = prediction_Upload)
predictionImage.pack(padx = 5, pady = 5)

# After Image frame

After_Image_Button = Button(File_upload_frame, width = 5, text='>', command = after_image)
After_Image_Button.pack(side = "left", fill = "y")



# 저장 경로 프레임(폴더 선택)
path_frame = LabelFrame(root, text = "저장 경로")
path_frame.pack(fill = "x", padx = 5, pady = 5)

txt_dest_path = Entry(path_frame)
txt_dest_path.pack(side = "left", fill = "x", expand = True)

btn_dest_path = Button(path_frame, text = "Folder", width = 10, command = destination_path)
btn_dest_path.pack(side = "right")


# 실행 프레임
frame_run = Frame(root)
frame_run.pack(fill = "x")

btn_close = Button(frame_run, padx = 5, pady = 5, text = "닫기", width = 12, command = root.quit)
btn_close.pack(side = "right", padx = 5, pady = 5)

btn_start = Button(frame_run, padx = 5, pady = 5, text = "시작", width = 12, command = start)
btn_start.pack(side = "right", padx = 5, pady = 5)

DSC_Score_Box = LabelFrame(frame_run, text = "DSC Score")



root.mainloop()
