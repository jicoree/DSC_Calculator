import cv2, os
import pydicom
import numpy as np
import matplotlib.pyplot as plt


currentpath = os.getcwd()
print(currentpath)

def dice(annotation, prediction):
        
    annotation = np.asarray(annotation).astype(np.bool_)
    prediction = np.asarray(prediction).astype(np.bool_)

    if annotation.shape != prediction.shape:
        raise ValueError("Shape mismatch: annotation and prediction must have the same shape.")

    intersection = np.logical_and(annotation, prediction)

    return np.sum(2. * intersection.sum()) / (np.sum(annotation.sum()) + np.sum(prediction.sum()))

def viz_all(input_img, gt_mask, pred_mask,
            save_to='./output', fn='sample',
            img_cm='gray', gt_cm='Blues', pred_cm='Reds',
            use_contour=False,
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
    
    os.makedirs(save_to, exist_ok=True)
   
    plt.figure(figsize=(11.69, 8.27), dpi = 300)
    plt.suptitle(str(filename + " " * 25 + "Dice score: {}".format(round(dice_score,4))), fontsize = 15, x = 0.5, y = 1, fontweight = 'bold')
    # plt.suptitle('PROS-CXR-01-CT-01', fontsize = 22, x = 0.5, y = 1, fontweight = 'bold')
    # plt.suptitle('Test', fontsize = 22, x = 0.8, y = 1, fontweight = 'bold')
    
    plt.subplot(2,3,1).set_axis_off()
    plt.imshow(input_img, cmap=img_cm, alpha = 1)
    plt.title("Input", size=12)

    plt.subplot(2,3,2).set_axis_off()
    plt.imshow(gt_mask, cmap=gt_cm)
    plt.title("Mask-Ground Truth", size=12)

    plt.subplot(2,3,3).set_axis_off()
    plt.imshow(pred_mask, cmap=pred_cm)
    plt.title("Mask-Prediction", size=12)

    plt.subplot(2,3,4).set_axis_off()
    plt.imshow(input_img, cmap=img_cm, alpha = 1)
    plt.imshow(np.logical_or(gt_mask, pred_mask), cmap = 'Greens', alpha=0.7)
    plt.imshow(np.logical_and(gt_mask,pred_mask), cmap = 'Purples', alpha=0.7)
    plt.title("Overlay-All", size=12)

    plt.subplot(2,3,5).set_axis_off()
    plt.imshow(input_img, cmap=img_cm, alpha = 1)
    plt.imshow(gt_mask, cmap=gt_cm, alpha=0.7)
    plt.title("Overlay-Ground Truth", size=12)

    plt.subplot(2,3,6).set_axis_off()
    plt.imshow(input_img, cmap=img_cm, alpha = 1) 
    plt.imshow(pred_mask, cmap=pred_cm, alpha=0.7)
    plt.title("Overlay-Prediction", size=12)

    
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
    
    plt.savefig(os.path.join(save_to, str(fn)+'_'+str(round(dice_score, 4))+'.pdf'), bbox_inches='tight')


path = "./"
file_list = sorted(os.listdir(path+'DCM'))
DCM_Extention = ['.dcm', '.DCM', '.Dcm', '.dCm', '.dcM', '.DCm', '.DcM', '.dCM']
# DCM_list = [file for file in file_list if file.endswith('dcm')]
DCM_list = [file for file in file_list if file.endswith(tuple(DCM_Extention))]

for i in range (len(DCM_list)):
    DCM_file = DCM_list[i]
    Output_name = os.path.basename(DCM_file)
    print(Output_name)
    filename, ext = os.path.splitext(Output_name)
    print(filename)    

    DCMimage = pydicom.read_file('./DCM/' + DCM_file, force = True).pixel_array

    annotation = cv2.imread('./annotation/' + filename + '.png', cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread('./prediction/' + filename + '.png', cv2.IMREAD_GRAYSCALE)


    dice_score = dice(annotation, prediction)
    print("Dice Coefficient is: {}".format(round(dice_score, 4)))

    viz_all(DCMimage, annotation, prediction,
            save_to='./output', fn=filename,
            img_cm='gray', gt_cm='Blues', pred_cm='Reds',
            use_contour=False,
            use_grid=False, grid_alpha=0.5)

