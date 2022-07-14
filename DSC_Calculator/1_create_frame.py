from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import os
import cv2
import pydicom
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


root = Tk()
root.title("DSC Calculator")
root.resizable(False, False)

def DCMUploadBtn():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("DCM 파일", "*.DCM"), ("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    for file in files:
        ds = pydicom.read_file[file]
        img = ds.pixel_array[file]
        cv2.imwrite(file + file.replace(".dcm", ".png"), img)
        
        # dcm_list_file.insert(END, file)
    pass

def annotation_Upload():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    # 사용자가 선택한 파일 목록
    for file in files:
        annotation_list_file.insert(END, file)

def prediction_Upload():
    files = filedialog.askopenfilenames(title = "파일을 선택하세요.", filetypes = (("PNG 파일", "*.png"), ("모든 파일", "*.*")))
    
    # 사용자가 선택한 파일 목록
    for file in files:
        prediction_list_file.insert(END, file)

# def del_file():
#     for index in reversed(list_file.curselection()):
#         list_file.delete(index)
        
def destination_path():
    folder_selected = filedialog.askdirectory()
    if folder_selected is None: # 사용자가 취소 누를 시
        return
    txt_dest_path.delete(0, END)
    txt_dest_path.insert(0, folder_selected)
    
def start():
    if annotation_list_file == 0:
        msgbox.showwarning("경고", "마스크 파일을 추가하세요.")
        return
    if prediction_list_file == 0:
        msgbox.showwarning("경고", "마스크 파일을 추가하세요.")
        return
    pass    

def before_image():
    pass

def after_image():
    pass


# File upload Frame

File_upload_frame = LabelFrame(root)
File_upload_frame.pack()

# Before Image frame

Before_Image_Button = Button(File_upload_frame, width = 5, text='<', command = before_image)
Before_Image_Button.pack(side = "left", fill = "y")


# DCM file 업로드 frame

DCM_frame = LabelFrame(File_upload_frame, text = "DICOM")
DCM_frame.pack(side = "left", padx = 5, pady = 5)

DCMImage_frame = Frame(DCM_frame, width = 400, height = 400, bg = "white")
DCMImage_frame.pack(padx = 5, pady = 5)

DCMImage = Button(DCM_frame, width = 20, height = 1, text = "Upload", command = DCMUploadBtn)
DCMImage.pack(padx = 5, pady = 5)


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



root.mainloop()
