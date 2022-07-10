from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import os
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *
from tkinter import filedialog

root = Tk()
root.title("DSC Calculator")
root.resizable(False, False)

def DCMUploadBtn():
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

# def predictionUplloadBtn():
#     pass


# DCM file 업로드 frame
File_upload_frame = LabelFrame(root)
File_upload_frame.pack()

DCM_frame = LabelFrame(File_upload_frame, text = "DICOM upload")
DCM_frame.pack(side = "left", padx = 5, pady = 5)


DCMlist_frame = Frame(DCM_frame)
DCMlist_frame.pack(padx = 5, pady = 5)

scrollbar = Scrollbar(DCMlist_frame)
scrollbar.pack(side = "right", fill = "y")

DCM_list_file = Listbox(DCMlist_frame, selectmode = "extended", height = 15, yscrollcommand = scrollbar.set)
DCM_list_file.pack(side = "left", fill = "both", expand = True)
scrollbar.config(command = DCM_list_file.yview)

DCMImage = Button(DCM_frame, width = 20, height = 1, text = "Upload", command = DCMUploadBtn)
DCMImage.pack(padx = 5, pady = 5)


# annotation 업로드 frame


annotation_frame = LabelFrame(File_upload_frame, text = "Annotation Mask upload")
annotation_frame.pack(side = "left", padx = 5, pady = 5)

annotationList_frame = Frame(annotation_frame)
annotationList_frame.pack(padx = 5, pady = 5)

scrollbar = Scrollbar(annotationList_frame)
scrollbar.pack(side = "right", fill = "y")

annotation_list_file = Listbox(annotationList_frame, selectmode = "extended", height = 15, yscrollcommand = scrollbar.set)
annotation_list_file.pack(side = "left", fill = "both", expand = True)
scrollbar.config(command = annotation_list_file.yview)

annotationImage = Button(annotation_frame, width = 20, height = 1, text = "Upload", command = annotation_Upload)
annotationImage.pack(padx = 5, pady = 5)



# prediction 업로드 frame

prediction_frame = LabelFrame(File_upload_frame, text = "Prediction Mask upload")
prediction_frame.pack(side = "left", padx = 5, pady = 5)

predictionList_frame = Frame(prediction_frame)
predictionList_frame.pack(padx = 5, pady = 5)

scrollbar = Scrollbar(predictionList_frame)
scrollbar.pack(side = "right", fill = "y")

prediction_list_file = Listbox(predictionList_frame, selectmode = "extended", height = 15, yscrollcommand = scrollbar.set)
prediction_list_file.pack(side = "left", fill = "both", expand = True)
scrollbar.config(command = prediction_list_file.yview)

predictionImage = Button(prediction_frame, width = 20, height = 1, text = "Upload", command = prediction_Upload)
predictionImage.pack(padx = 5, pady = 5)


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
