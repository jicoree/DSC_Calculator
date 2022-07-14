import cv2, os, argparse

import pandas as pd

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt



'''
make_data_list() -> 한장씩 추론하는 경우 사용하지 않음.
'''
def make_data_list(a, dir):
    return [os.path.join(a, dir, file) for file in os.listdir(os.path.join(a, str(dir)))]

'''
make_data_df_gt() -> 한장씩 추론하는 경우 사용하지 않음.
'''
def make_data_df_gt(data_path):
    img = make_data_list(data_path, 'dicom')
    mask = make_data_list(data_path, 'mask')
    df = pd.DataFrame(list(zip(sorted(img), sorted(mask))),
                      columns=['image_filename', 'mask_filename'])
    df.info()
    return df

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
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "PROS-CXR-01-CT-01")
    
    parser.add_argument("-d", "--dicom_img", type=str, help="dicom 이미지 파일 경로",
                        default="./test/dicom/image.png")
    
    parser.add_argument("-g", "--ground_truth_mask", type=str, help="정답 마스크 파일 경로",
                        default="./test/dicom/mask.png")
    
    parser.add_argument("-p", "--prediction_mask", type=str, help="PROS CXR:01 예측 마스크 파일 경로",
                        default="./test/aquamarine/pred.png")
    
    parser.add_argument("-s", "--save_to", type=str, help="최종 결과 저장 경로",
                        default="./output")
    
    opt = parser.parse_args()
    print(opt)
    
    '''
    사용 예시:
    
    -d ./abnormal/dicom/0022_17953234_11\(pneumothorax\)0_1000000.png
    -g ./abnormal/mask/0022_17953234_11\(pneumothorax\)0_1000000.tif
    -p ./abnormal_predmask/5.png
    -s ./output
    '''
    
    image_path = opt.dicom_img
    gt_mask_path = opt.ground_truth_mask
    pred_mask_path = opt.prediction_mask
    
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
            save_to=opt.save_to, fn=file_name,
            img_cm='gray', gt_cm='Greens', pred_cm='Reds',
            use_contour=False,
            use_grid=True, grid_alpha=0.5)