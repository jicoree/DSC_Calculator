from http.client import REQUESTED_RANGE_NOT_SATISFIABLE

import cv2, os
import numpy as np

import tkinter.ttk as ttk
import tkinter.messagebox as msgbox

from tkinter import *
from tkinter import filedialog
from tkinter import ttk

from PIL import Image, ImageTk

import matplotlib.pyplot as plt

import pydicom




def dice(annotation, prediction):
        

    annotation = np.asarray(annotation).astype(np.bool_)
    prediction = np.asarray(prediction).astype(np.bool_)

    if annotation.shape != prediction.shape:
        raise ValueError("Shape mismatch: annotation and prediction must have the same shape.")


    intersection = np.logical_and(annotation, prediction)

    return (2. * intersection.sum()) / (annotation.sum() + prediction.sum()) 


def viz_all(input_img, gt_mask, pred_mask,
            save_to='./output', fn='sample',
            img_cm='gray', gt_cm='Blues', pred_cm='Reds',
            use_contour=True,
            use_grid=False, grid_alpha=0.5
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
    
    
    dice_score = dice(annotation, prediction)
    # print("Dice score: {}".format(round(dice_score,4)))
    
    os.makedirs(save_to, exist_ok=True)
   
    plt.figure(figsize=(30, 20))
    plt.suptitle(str("Dice score: {}".format(round(dice_score,4))), fontsize = 22, x = 0.1, y = 1, fontweight = 'bold')


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
    plt.imshow(gt_mask, cmap=gt_cm, alpha=0.5)
    plt.imshow(pred_mask, cmap=pred_cm, alpha=0.5)
    plt.title("Overlay-All", size=22)

    plt.subplot(2,3,5)
    plt.imshow(input_img, cmap=img_cm)
    plt.imshow(gt_mask, cmap=gt_cm, alpha=0.5)
    plt.title("Overlay-Ground Truth", size=22)

    plt.subplot(2,3,6)
    plt.imshow(input_img, cmap=img_cm) 
    plt.imshow(pred_mask, cmap=pred_cm, alpha=0.5)
    plt.title("Overlay-Prediction", size=22)

    
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
    
    plt.savefig(os.path.join(save_to, str(dice_score)+'_'+fn+'.png'), bbox_inches='tight')

def destination_path():
    folder_selected = filedialog.askdirectory()
    if folder_selected is None: # 사용자가 취소 누를 시
        return
    txt_dest_path.delete(0, END)
    txt_dest_path.insert(0, folder_selected)


def start(DCMimage, annotation, prediction):
    dice_score = dice(annotation, prediction)
    print("Dice Coefficient is: {}".format(round(dice_score, 4)))

    viz_all(DCMimage, annotation, prediction,
            save_to='F:/ETC/DSC_Calculator/DSC_Calculator/output', fn='output',
            img_cm='gray', gt_cm='Blues', pred_cm='Reds',
            use_contour=True,
            use_grid=False, grid_alpha=0.5)
    

# def DCMUploadBtn():
#     global DCMimage # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
#     DCMimage = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(("DCM 파일", "*.DCM"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")))
#     DCMimage = plt.imshow(pydicom.read_file(DCMimage).pixel_array, cmap = 'gray')
#     Label(DCMImage_frame, image = DCMimage).pack() # 파일경로 view

def annotation_Upload():
    global annotation
    global annotation_viewer
    
    annotation = filedialog.askopenfilename(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    annotation_viewer = ImageTk.PhotoImage(file = annotation)
    ttk.Label(annotationImage_frame, image = annotation_viewer).pack()
    return print(annotation)
    # annotationViewer = Label(annotation_frame, image = annotation).pack()
            

def prediction_Upload():
    global prediction
    global prediction_viewer
    prediction = filedialog.askopenfilename(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    prediction_viewer = ImageTk.PhotoImage(file = prediction)
    ttk.Label(predictionImage_frame, image = prediction_viewer).pack()
    return prediction
    # annotationViewer = Label(annotation_frame, image = annotation).pack()

root = Tk()
root.title("DSC Calculator")
# root.resizable(False, False)



# File upload Frame

File_upload_frame = LabelFrame(root)
File_upload_frame.pack()

# Before Image frame

# Before_Image_Button = Button(File_upload_frame, width = 5, text='<', command = before_image)
# Before_Image_Button.pack(side = "left", fill = "y")


# DCM file 업로드 frame


# DCM_frame = LabelFrame(File_upload_frame, text = "DICOM")
# DCM_frame.pack(side = "left", padx = 5, pady = 5)

# DCMImage_frame = Frame(DCM_frame, width = 400, height = 400, bg = "white")
# DCMImage_frame.pack(padx = 5, pady = 5)

# DCMImage = Button(DCM_frame, width = 20, height = 1, text = "Upload", command = DCMUploadBtn)
# DCMImage.pack(padx = 5, pady = 5)



# annotation 업로드 frame


annotation_frame = LabelFrame(File_upload_frame, text = "Annotation")
annotation_frame.pack(side = "left", padx = 5, pady = 5)

annotationImage_frame = Frame(annotation_frame, width = 400, height = 400, bg = "white")
annotationImage_frame.pack(padx = 5, pady = 5)

annotationImage = Button(annotation_frame, width = 20, height = 1, text = "Upload", command = annotation_Upload)
annotationImage.pack(padx = 5, pady = 5)


# prediction 업로드 frame

prediction_frame = LabelFrame(File_upload_frame, text = "Prediction")
prediction_frame.pack(side = "left", padx = 5, pady = 5)

predictionImage_frame = Frame(prediction_frame, width = 400, height = 400, bg = "white")
predictionImage_frame.pack(padx = 5, pady = 5)

predictionImage = Button(prediction_frame, width = 20, height = 1, text = "Upload", command = prediction_Upload)
predictionImage.pack(padx = 5, pady = 5)


# After Image frame

# After_Image_Button = Button(File_upload_frame, width = 5, text='>', command = after_image)
# After_Image_Button.pack(side = "left", fill = "y")



# 저장 경로 프레임(폴더 선택)
path_frame = LabelFrame(root, text = "저장 경로")
path_frame.pack(fill = "x", padx = 5, pady = 5)

txt_dest_path = Entry(path_frame)
txt_dest_path.pack(side = "left", fill = "x", expand = True)

btn_dest_path = Button(path_frame, text = "Folder", width = 10, command = destination_path)
btn_dest_path.pack(side = "right")


# # 실행 프레임
frame_run = Frame(root)
frame_run.pack(fill = "x")

# btn_close = Button(frame_run, padx = 5, pady = 5, text = "닫기", width = 12, command = root.quit)
# btn_close.pack(side = "right", padx = 5, pady = 5)

btn_start = Button(frame_run, padx = 5, pady = 5, text = "시작", width = 12, command = dice)
btn_start.pack(side = "right", padx = 5, pady = 5)

print(annotation)

root.mainloop()



