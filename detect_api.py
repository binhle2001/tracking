from collections import OrderedDict
from tkinter import Tk, filedialog

from draw_plate import overlay_plates_on_frame
from boxmot import  OCSORT
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import cv2
import json
import torch
import re
from queue_dict import pop
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import math
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import  LoadImages, LoadImagesBitmap, LoadImagesFromListBitmap
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, is_ascii, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
from shapely.geometry import Polygon, box

import subprocess as sp
import psutil

import ctypes as C
import cv2

with open('config.json') as f:
    config = json.load(f)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def select_files():
    """Hiển thị hộp thoại để người dùng chọn nhiều tệp ảnh."""
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_paths = filedialog.askopenfilenames(
        title="Chọn tệp",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.mp4")]
    )
    return file_paths

def is_nested(rect1, rect2):
    """
    Kiểm tra xem một hình chữ nhật có nằm hoàn toàn bên trong hình chữ nhật kia không.
    Mỗi hình chữ nhật được định nghĩa bằng (x_min, y_min, x_max, y_max).
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    # Kiểm tra rect1 có nằm trong rect2 không
    rect1_in_rect2 = (x1_min >= x2_min and y1_min >= y2_min and 
                      x1_max <= x2_max and y1_max <= y2_max)
    
    # Kiểm tra rect2 có nằm trong rect1 không
    rect2_in_rect1 = (x2_min >= x1_min and y2_min >= y1_min and 
                      x2_max <= x1_max and y2_max <= y1_max)
    
    return rect1_in_rect2 or rect2_in_rect1


def phanTramGiaoNhau(mangHang1, mangHang1_):
    try:
        x1_ = mangHang1_[1]
        y1_ = mangHang1_[2]
        x2_ = x1_ + mangHang1_[3]
        y2_ = y1_
        x3_ = x1_ + mangHang1_[3]
        y3_ = y1_ + mangHang1_[4]
        x4_ = x1_
        y4_ = y3_
        box_1 = [[x1_, y1_], [x2_, y2_],
                 [x3_, y3_], [x4_, y4_]]

        x1 = mangHang1[1]
        y1 = mangHang1[2]
        x2 = x1 + mangHang1[3]
        y2 = y1
        x3 = x1 + mangHang1[3]
        y3 = y1 + mangHang1[4]
        x4 = x1
        y4 = y3
        box_2 = [[x1, y1], [x2, y2],
                 [x3, y3], [x4, y4]]

        # rect1 = box(x4_, y4_, x2_, y2_)
        # rect2 = box(x4, y4, x2, y2)
        # intersection = rect1.intersection(rect2)
        # print(intersection)
        # iou1 = intersection.area / rect1.area
        # iou2 = intersection.area / rect2.area
        # iou = iou1
        # if iou2 > iou1:
        #     iou = iou2
        # return iou

        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        # print(poly_1)
        # print(poly_2)
        # print(poly_1.intersection(poly_2))

        iou1 = poly_1.intersection(poly_2).area / poly_1.area
        iou2 = poly_1.intersection(poly_2).area / poly_2.area

        iou = iou1
        if iou2 > iou1:
            iou = iou2
        return iou
    except Exception as e:
        print('loi phan tram giao nhau')
        print(e)
        return 0
    
def checkPlateNumberFormat_VN(plateNumber):
    try:
        if(plateNumber != ''):
            #1. car: 30A-1234, 30A-12345
            # pattern = re.compile("^[0-9]{2}[A-Z]{1}-[0-9]{4,5}$")
            pattern = re.compile("^[0-9]{2}[A-Z]{1}-[0-9]{5}$")            
            if(pattern.match(plateNumber)):
                return True

            #2. motor: 30-A1-1234, 30-A1-12345
            pattern = re.compile("^[0-9]{2}-[A-Z]{1}[1-9]{1}-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #3. car LD: 30LD-1234, 30LD-12345 Xe của các doanh nghiệp có vốn nước ngoài, xe thuê của nước ngoài, xe của Công ty nước ngoài trúng thầu có ký hiệu "LD"
            # pattern = re.compile("^[0-9]{2}LD-[0-9]{4,5}$")
            pattern = re.compile("^[0-9]{2}LD-[0-9]{5}$")            
            if(pattern.match(plateNumber)):
                return True

            #4. car QĐ: KP-34-56
            pattern = re.compile("^[A-Z]{2}-[0-9]{2}-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #5. motor MĐ: 30-MD1-12345 Xe máy điện có ký hiệu "MĐ";
            pattern = re.compile("^[0-9]{2}-MD[1-9]{1}-[0-9]{5}$")
            if(pattern.match(plateNumber)):
                return True

            #6. motor 50cc: 30-AB-1234, 30-AB-12345
            pattern = re.compile("^[0-9]{2}-[A-Z]{2}-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #7. car NN: 29-636-NN-11
            pattern = re.compile("^[0-9]{2}-[0-9]{3}-NN-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #8. car NN old: 29-NN-636-11
            pattern = re.compile("^[0-9]{2}-NN-[0-9]{3}-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #9. car NG: 29-636-NG-11
            pattern = re.compile("^[0-9]{2}-[0-9]{3}-NG-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #10. car NG old: 29-NG-636-11
            pattern = re.compile("^[0-9]{2}-NG-[0-9]{3}-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #11. car QT: 29-636-QT-11
            pattern = re.compile("^[0-9]{2}-[0-9]{3}-QT-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #12. car QT old: 29-QT-636-11
            pattern = re.compile("^[0-9]{2}-QT-[0-9]{3}-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #13. car CV: 29-636-CV-11
            pattern = re.compile("^[0-9]{2}-[0-9]{3}-CV-[0-9]{2}$")
            if(pattern.match(plateNumber)):
                return True

            #14. KT: 30KT-12345 Xe Quân đội làm kinh tế có ký hiệu "KT";
            pattern = re.compile("^[0-9]{2}KT-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #15. LA: 30LA-12345
            pattern = re.compile("^[0-9]{2}LA-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #16. LB: 30LB-12345
            pattern = re.compile("^[0-9]{2}LB-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #17. DA: 30DA-12345 Xe của các Ban quản lý dự án do nước ngoài đầu tư có ký hiệu "DA"
            # pattern = re.compile("^[0-9]{2}DA-[0-9]{4,5}$")
            pattern = re.compile("^[0-9]{2}DA-[0-9]{5}$")            
            if(pattern.match(plateNumber)):
                return True

            #18. HC: 30HC-12345 - Ô tô phạm vi hoạt động hạn chế có ký hiệu "HC"
            pattern = re.compile("^[0-9]{2}HC-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #19. TD: 30TD-12345 - Xe cơ giới sản xuất, lắp ráp trong nước, được Thủ tướng Chính phủ cho phép triển khai thí điểm có ký hiệu "TĐ"
            pattern = re.compile("^[0-9]{2}TD-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #20. MK: 30MK-12345 Máy kéo có ký hiệu "MK"
            pattern = re.compile("^[0-9]{2}MK-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

            #21. CD: 30CD-12345 - Xe chuyên dùng của lực lượng Công an nhân dân sử dụng vào mục đích an ninh: Biển số nền màu xanh, chữ và số màu trắng có ký hiệu "CD"
            pattern = re.compile("^[0-9]{2}CD-[0-9]{4,5}$")
            if(pattern.match(plateNumber)):
                return True

    except Exception as e:
        print('checkPlateNumberFormat_VN() error: ')
        print(e)
    return False




def checkBienSo2(source, model_nd, names_nd, device, recs_Crop, confident_bs = 0):
    tic = time_sync()
    list_traloi = []
    try:
        imgsz_nd=[320, 320]
        stride_nd = 32
        pt_nd = True

        tic = time.time()
        dataset_BS = LoadImagesFromListBitmap(source, img_size=imgsz_nd, stride=stride_nd, auto=False)
 
        for path, im, im0s, vid_cap, s in dataset_BS:          
            # print(im.shape)
            # print(len(im))
            tic = time.time()
            im = torch.from_numpy(im).to(device)
            # im = im.half() if model_nd.fp16 else im.float()  # uint8 to fp16/32
            im = im.half()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model_nd(im)
            # NMS
            pred = non_max_suppression(pred, 0.5, 0.45, None, False, max_det=100)
            for i, det in enumerate(pred):  # per image  
                bienso = ''
                # -1 xe may dien, 0-xe gan may, 1-xe moto, 2-xe oto
                loaixe = 1
                # 0-biển trắng 1-biển xanh 2-biển đỏ 3-biển vàng
                maubien = 0
                gocnghieng = 0
                # gocnghieng am la nghieng ben phai, duong la nghieng ben trai

                p, im0, frame = path, im0s[i].copy(), dataset_BS.count            
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                height0 = im0.shape[0]
                width0 = im0.shape[1]
                index = 0
                if len(det):
                    toc = time.time()
                    # mảng xử lý
                    mangXuLy = np.array([0, 0, 0, 0, 0, 0], dtype=int)
                    mangXuLy_khong_co1 = np.array([0, 0, 0, 0, 0, 0], dtype=int)

                    # print(det)
                    # Rescale boxes from img_size to im0 size                    
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    isCoKyTu = False
                    isCoKyTuKhac1 = False
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # conf_ = int(float(conf) * 100 + 0.5)
                        confidence = float(conf)
                        c = int(cls)  # integer class
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                gn).view(-1).tolist()  # normalized xywh
                        # xy la toa do diem tam
                        width = xywh[2] * width0
                        height = xywh[3] * height0
                        x_tam = xywh[0] * width0
                        y_tam = xywh[1] * height0

                        x_min = (x_tam - width/2)
                        y_min = (y_tam - height/2)
                        x_max = (x_tam + width/2)
                        y_max = (y_tam + height/2)
                        nguong = 0.6
                        if c == 32:
                            if confidence <= nguong or 2*x_min >= width0 or 2*x_max <= width0 or 3*(x_max - x_min) <= width0:
                                maubien = 0
                                # print('mau bien chua dung')
                            else:
                                maubien = 1
                                # print("xanh " + str(confidence))
                        elif c == 33:
                            if confidence <= nguong or 2*x_min >= width0 or 2*x_max <= width0 or 3*(x_max - x_min) <= width0:
                                maubien = 0
                                # print('mau bien chua dung')
                            else:
                                maubien = 3
                                # print("vang " + str(confidence))
                        elif c == 34:
                            if confidence <= nguong or 2*x_min >= width0 or 2*x_max <= width0 or 3*(x_max - x_min) <= width0:
                                maubien = 0
                                # print('mau bien chua dung')
                            else:
                                maubien = 2
                                # print("do " + str(confidence))
                        elif c < 32:
                            if confidence >= nguong:
                                isCoKyTu = True
                                conf_ = int(confidence * 100 + 0.5)
                                mang = np.array([c, int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min), conf_], dtype=int)
                                mangXuLy = np.vstack([mangXuLy, mang])
                                if c != 1:
                                    isCoKyTuKhac1 = True
                                    mangXuLy_khong_co1 = np.vstack([mangXuLy_khong_co1, mang])

                    trungvi_w = 0
                    trungvi_h = 0

                    if isCoKyTu and len(mangXuLy.shape) == 2:
                        mangXuLy = np.delete(mangXuLy, (0), axis=0)
                        if isCoKyTuKhac1 and len(mangXuLy_khong_co1.shape) == 2:
                            mangXuLy_khong_co1 = np.delete(mangXuLy_khong_co1, (0), axis=0)
                            try:
                                # print(mangXuLy)
                                if len(mangXuLy_khong_co1) > 1:

                                    trungvi_w = np.median(mangXuLy_khong_co1[:, 3])
                                    trungvi_h = np.median(mangXuLy_khong_co1[:, 4])
                                elif len(mangXuLy_khong_co1) == 1:

                                    trungvi_w = mangXuLy_khong_co1[:, 3]
                                    trungvi_h = mangXuLy_khong_co1[:, 4]
                                # loai bo ky tu co do rong > 1.5*trungvi_w
                                if trungvi_w > 0 and trungvi_h > 0:

                                    list_idx_loaiBo = []
                                    for idx, val in enumerate(mangXuLy):
                                        isLoaiBo = False
                                        # Loại bỏ trường hợp có width rộng va nho
                                        if val[0] != 1:
                                            if (val[3] > 1.5 * trungvi_w) or (val[3] < 0.5 * trungvi_w):
                                                isLoaiBo = True
                                                list_idx_loaiBo.append(idx)

                                            # elif (val[4] > 1.3334 * trungvi_h) or (val[4] < 0.6667 * trungvi_w):
                                            #     isLoaiBo = True
                                            #     list_idx_loaiBo.append(idx)
                                    if len(list_idx_loaiBo) > 0:
                                        count = 0
                                        for idx, val in enumerate(mangXuLy):
                                            try:
                                                if idx in list_idx_loaiBo:
                                                    mangXuLy = np.delete(mangXuLy, (idx - count), axis=0)
                                                    count = count + 1

                                            except Exception as e:
                                                print("loi xoa " + str(e))

                                    # xap sep theo thu tu chieu x
                                    count = 0
                                    # print(mangXuLy)
                                    if len(mangXuLy) >= 1 and len(mangXuLy) <= 20:
                                        # print("mang xu ly :,1")
                                        # print(mangXuLy[:, 1])
                                        mangXuLy_sort_x = np.argsort(mangXuLy[:, 1])
                                        # print("mangXuLy_sort_x")
                                        # print(mangXuLy_sort_x)
                                        is2Hang = False
                                        bienso1hang = ''
                                        for idx, val in enumerate(mangXuLy_sort_x):
                                            # bienso1hang += str(names_nd[mangXuLy[val, 0]])
                                            if idx + 1 < len(mangXuLy_sort_x):
                                                delda_y = abs(mangXuLy[val, 2] -
                                                              mangXuLy[mangXuLy_sort_x[idx + 1], 2])
                                                if 2 * delda_y > trungvi_h:
                                                    # print(str(val) + "  " + str(delda_y) +
                                                    #       " " + str(trungvi_h))
                                                    count += 1
                                                    if count >= 2:
                                                        is2Hang = True
                                                        break
                                        # print('bien 2 hang: ' + str(is2Hang))

                                        if is2Hang == False:
                                            xmin_ymin_BienSo = []
                                            # BIEN SO 1 HANG
                                            loaixe = 2
                                            mangHang1_bs1Hang = mangXuLy[0]
                                            current_NoiDung_Yolo = ""

                                            for idx, val in enumerate(mangXuLy_sort_x):
                                                if len(bienso1hang) == 0:
                                                    mangHang1_bs1Hang = mangXuLy[val]
                                                    bienso1hang += str(names_nd[mangXuLy[val, 0]])
                                                    xmin_ymin_BienSo.append([mangXuLy[val, 1], mangXuLy[val, 2]])
                                                else:
                                                    # kiem tra trung khop voi ky tu truoc
                                                    mangHang1_bs1Hang_ = mangXuLy[val]
                                                    phantram = phanTramGiaoNhau(
                                                        mangHang1_bs1Hang, mangHang1_bs1Hang_)
                                                    # print(phantram)
                                                    if (phantram > 0.8):
                                                        if mangHang1_bs1Hang[5] < mangHang1_bs1Hang_[5]:
                                                            # doi lai gia tri
                                                            mangHang1_bs1Hang = mangHang1_bs1Hang_
                                                            bienso1hang = bienso1hang[0:len(
                                                                bienso1hang) - 1] + str(names_nd[mangHang1_bs1Hang_[0]])
                                                    # loai bo truong hop doc sai 2 so 1 o canh nhau
                                                    elif (phantram >= 0.5) and (mangHang1_bs1Hang_[0] == 1) and (mangHang1_bs1Hang[0] == 1):
                                                        # bo qua so 1 o day
                                                        print('so 1 o vi tri khong dung')
                                                    else:
                                                        # khong trung thi them moi
                                                        mangHang1_bs1Hang = mangXuLy[val]
                                                        bienso1hang += str(names_nd[mangXuLy[val, 0]])
                                                        xmin_ymin_BienSo.append([mangXuLy[val, 1], mangXuLy[val, 2]])
                                            # print(bienso1hang)

                                            # phân loại biển đỏ
                                            if (maubien == 2):
                                                if (len(bienso1hang) == 6) and (bienso1hang[0: 2].isdigit() == False):
                                                    # if maubien == 0:
                                                    #     maubien = 2
                                                    kytu0 = bienso1hang[0]
                                                    kytu1 = bienso1hang[1]

                                                    if kytu0 == '0':
                                                        kytu0 = 'Q'
                                                    elif kytu0 == '8':
                                                        kytu0 = 'B'
                                                    elif kytu0 == '7':
                                                        kytu0 = 'V'
                                                    elif kytu0 == '1':
                                                        kytu0 = 'T'
                                                    if kytu1 == '0':
                                                        kytu1 = 'Q'
                                                    elif kytu1 == '8':
                                                        kytu1 = 'B'
                                                    elif kytu1 == '1':
                                                        kytu1 = 'T'

                                                    bienso = kytu0 + kytu1 + '-' + \
                                                        bienso1hang[2: 4] + '-' + bienso1hang[4: 6]
                                                elif len(bienso1hang) < 6:
                                                    bienso = ''
                                                elif len(bienso1hang) > 6:
                                                    kytu0 = bienso1hang[0]
                                                    kytu1 = bienso1hang[1]
                                                    countchucai = 0
                                                    # countchuso = 0
                                                    for kytu in bienso1hang:
                                                        if kytu.isdigit() == False:
                                                            countchucai = countchucai + 1
                                                    #         countchuso = countchuso + 1
                                                    #     else:
                                                    #         countchucai = countchucai + 1
                                                    # if countchucai >= 4 or countchuso > 6:
                                                    #     # nham vao chu cai
                                                    #     bienso = ''
                                                    if countchucai == 2 and kytu0.isdigit() == False and kytu1.isdigit() == False:
                                                        bienso = kytu0 + kytu1 + '-' + \
                                                            bienso1hang[2: 4] + '-' + bienso1hang[4: 6]
                                                    else:
                                                        bienso = ''

                                            # elif (len(bienso1hang) > 6):
                                            elif len(bienso1hang) >= 7:

                                                if ('NN' in bienso1hang) or ('NG' in bienso1hang) or ('HG' in bienso1hang) or ('CV' in bienso1hang) or ('QT' in bienso1hang) or (len(bienso1hang) == 9 and 'N6' in bienso1hang and bienso1hang.index('N6') == 5):
                                                    if ('NN' in bienso1hang):
                                                        # 29-NG-636-11   29-636-NG-11
                                                        index_char = bienso1hang.index('NN')
                                                        if index_char == 2:
                                                            bienso = bienso1hang[0:2] + '-NN-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        elif index_char == 5:
                                                            bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-NN-' + bienso1hang[7:]
                                                        else:
                                                            bienso = bienso1hang
                                                    elif ('NG' in bienso1hang):
                                                        # 29-NG-636-11   29-636-NG-11
                                                        index_char = bienso1hang.index('NG')
                                                        if index_char == 2:
                                                            bienso = bienso1hang[0:2] + '-NG-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        elif index_char == 5:
                                                            bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-NG-' + bienso1hang[7:]
                                                        else:
                                                            bienso = bienso1hang
                                                    elif ('HG' in bienso1hang):
                                                        # 29-NG-636-11   29-636-NG-11
                                                        index_char = bienso1hang.index('HG')
                                                        if index_char == 2:
                                                            bienso = bienso1hang[0:2] + '-NG-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        elif index_char == 5:
                                                            bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-NG-' + bienso1hang[7:]
                                                        else:
                                                            bienso = bienso1hang
                                                    elif ('N6' in bienso1hang):
                                                        # # 29-NG-636-11   29-636-NG-11
                                                        # index_char = bienso1hang.index('HG')
                                                        # if index_char == 2:
                                                        #     bienso = bienso1hang[0:2] + '-NG-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        # elif index_char == 5:
                                                        bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-NG-' + bienso1hang[7:]
                                                        # else:
                                                        #     bienso = bienso1hang
                                                    elif ('CV' in bienso1hang):
                                                        # 29-NG-636-11   29-636-NG-11
                                                        index_char = bienso1hang.index('CV')
                                                        if index_char == 2:
                                                            bienso = bienso1hang[0:2] + '-CV-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        elif index_char == 5:
                                                            bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-CV-' + bienso1hang[7:]
                                                        else:
                                                            bienso = bienso1hang
                                                    elif ('QT' in bienso1hang):
                                                        # 29-NG-636-11   29-636-NG-11
                                                        index_char = bienso1hang.index('QT')
                                                        if index_char == 2:
                                                            bienso = bienso1hang[0:2] + '-QT-' + bienso1hang[4:7] + '-' + bienso1hang[7:]
                                                        elif index_char == 5:
                                                            bienso = bienso1hang[0:2] + '-' + bienso1hang[2:5] + '-QT-' + bienso1hang[7:]
                                                        else:
                                                            bienso = bienso1hang
                                                # bien xe ô tô thông thường 30A1234 30LD1234 30LD12345 30NN12312
                                                # ky tu thu 4 la chu cai
                                                elif bienso1hang[3].isdigit() == False:
                                                    if bienso1hang[0].isdigit() and bienso1hang[1].isdigit():
                                                        # # bien 1 hang la Oto
                                                        # 30LD-12345
                                                        # if ('LD' in bienso1hang) or ('DA' in bienso1hang) or ('KT' in bienso1hang) or ('NN' in bienso1hang) or ('NG' in bienso1hang):
                                                        bienso = bienso1hang[0: 4] + '-'
                                                        phia_sau = bienso1hang[4:]
                                                        if len(phia_sau) > 5:
                                                            phia_sau = phia_sau[0:]
                                                        bienso += phia_sau
                                                    # else:
                                                    #     bienso = ''
                                                else:
                                                    # print('aaaaaaaaaaaaaaaaaaaa2')
                                                    countchucai = 0
                                                    for kytu_ in bienso1hang:
                                                        if kytu_.isdigit() == False:
                                                            countchucai = countchucai + 1
                                                    # 30A-12345
                                                    haiKyTuDau = bienso1hang[0:2]
                                                    kytu = bienso1hang[2]
                                                    if haiKyTuDau.isdigit():
                                                        if kytu == '8':
                                                            kytu = 'B'
                                                        elif kytu == '2':
                                                            kytu = 'Z'
                                                        # if countchucai == 0 or (countchucai > 0 and kytu.isdigit()):
                                                        #     bienso = ''
                                                        # else:
                                                        bienso = bienso1hang[0: 2] + kytu + '-'
                                                        phia_sau = bienso1hang[3:]
                                                        if len(phia_sau) > 5:
                                                            phia_sau = phia_sau[0: 5]
                                                        bienso += phia_sau
                                                    else:
                                                        bienso = bienso1hang

                                            # elif len(bienso1hang) > 9:
                                            #     bienso = ''
                                            else:
                                                bienso = bienso1hang

                                            # print(bienso)
                                            if len(bienso) > 0 and len(xmin_ymin_BienSo) >= 6:
                                                x_0 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 3][0]
                                                y_0 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 3][1]
                                                x_1 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 2][0]
                                                y_1 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 2][1]
                                                x_2 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 1][0]
                                                y_2 = xmin_ymin_BienSo[len(xmin_ymin_BienSo) - 1][1]

                                                # radians_01 = math.atan((y_1 - y_0) / (x_1 - x_0));
                                                # angle_01 = radians_01 * (180 / math.pi);
                                                radians_12 = math.atan((y_2 - y_1) / (x_2 - x_1));
                                                angle_12 = radians_12 * (180 / math.pi);
                                                if angle_12 != 0:
                                                    gocnghieng = -angle_12
                                        else:
                                            # bien 2 hang
                                            mangXuLy_sort_y = np.argsort(mangXuLy[:, 2])
                                            # tim list ky tu thuoc hang 1
                                            listHang1 = []
                                            listHang1_nghingo = []
                                            listKhongHang1 = []

                                            listHang1.append(mangXuLy_sort_y[0])
                                            y_min = mangXuLy[mangXuLy_sort_y[0], 2]

                                            y_max = mangXuLy[mangXuLy_sort_y[len(
                                                mangXuLy_sort_y) - 1], 2]
                                            # print('trungvi: ' + str(trungvi_h))
                                            # print('ymin ' + str(y_min))
                                            # print('ymax ' + str(y_max))

                                            # print(mangXuLy_sort_y)
                                            listDelta_y = []
                                            y_kytutruoc_ = 0
                                            x_kytutruoc_ = 0

                                            for idy, val in enumerate(mangXuLy_sort_y):
                                                # print('kytu ' + str(mangXuLy[val, 0]))
                                                if idy == 0:
                                                    listDelta_y.append(0)
                                                    listHang1.append(val)
                                                    y_kytutruoc_ = mangXuLy[val, 2]
                                                    x_kytutruoc_ = mangXuLy[val, 1]
                                                    # print("kytu 1: " + str(mangXuLy[val, 0]))
                                                elif idy < 7:
                                                    delda_y_min = abs(y_min - mangXuLy[val, 2])
                                                    delda_y_kytutruoc = abs(y_kytutruoc_ - mangXuLy[val, 2])
                                                    delda_x_kytutruoc = abs(x_kytutruoc_ - mangXuLy[val, 1])

                                                    # # print(str(delda_y_min) + "   " + str(trungvi_h))
                                                    # # print(mangXuLy[val, 2])
                                                    # # print(delda_y_min)
                                                    # # if delda_y_min < 0.7 * trungvi_h:
                                                    # print('kytu ' + str(mangXuLy[val, 0]))
                                                    # print('toa do y ' + str(delda_y_kytutruoc) + "  " + str(trungvi_h))
                                                    # print('toa do y_min ' + str(delda_y_min) + "  " + str(trungvi_h))
                                                    # print('toa do x ' + str(delda_x_kytutruoc) + "  " + str(trungvi_w))

                                                    isKyTuHang1 = False
                                                    if (delda_y_min < 0.5*trungvi_h) or ((delda_y_kytutruoc < 0.5 * trungvi_h) and (delda_x_kytutruoc <= 2 * trungvi_w)):
                                                        isKyTuHang1 = True
                                                        # print("val1 " + str(val))
                                                        # print("delda_y_min " + str(delda_y_min))
                                                        # print("0.5*trungvi_h " + str(0.5*trungvi_h))
                                                        # print("delda_y_kytutruoc " + str(delda_y_kytutruoc))
                                                        # print("delda_x_kytutruoc " + str(delda_x_kytutruoc))
                                                        # print("2 * trungvi_w " + str(2 * trungvi_w))
                                                    # xử lý ký tự chữ cai
                                                    elif (mangXuLy[val, 0] > 9) and (delda_x_kytutruoc < 3 * trungvi_w):
                                                        # truong hop xe 3 so cu
                                                        if delda_y_kytutruoc < 0.6 * trungvi_h:
                                                            isKyTuHang1 = True
                                                        # truong hop xe 4,5 so
                                                        elif (len(mangXuLy) > 7) and (delda_y_kytutruoc < 0.8 * trungvi_h):
                                                            isKyTuHang1 = True
                                                        # print("val2 " + str(val))
                                                        # print("delda_y_kytutruoc " + str(delda_y_kytutruoc))
                                                        # print("0.8*trungvi_h " + str(0.75*trungvi_h))
                                                        # print("mangXuLy[val, 0] " + str(mangXuLy[val, 0]))
                                                        # print("delda_x_kytutruoc " + str(delda_x_kytutruoc))
                                                        # print("3 * trungvi_w " + str(3 * trungvi_w))

                                                    # else:
                                                    #     print("kytu 2: " + str(mangXuLy[val, 0]))

                                                    if isKyTuHang1:
                                                        # print("kytu 1: " + str(mangXuLy[val, 0]))
                                                        listDelta_y.append(delda_y_min)
                                                        listHang1.append(val)
                                                        y_kytutruoc_ = mangXuLy[val, 2]
                                                        x_kytutruoc_ = mangXuLy[val, 1]

                                                # elif 4 * delda_y_min < 3 * trungvi_h:

                                            if y_min <= 1:
                                                # print('aaaaaaaaaaaaaaaaaa2')
                                                trungvi_deltay = np.median(listDelta_y)
                                                count_ = 0

                                                for id, val in enumerate(listDelta_y):
                                                    # print(str(val) + "    " + str(trungvi_h))
                                                    heso = 2.5
                                                    if trungvi_h < 45:
                                                        heso = 2.1
                                                    if heso*val >= trungvi_h:
                                                        # print('co')
                                                        # ky tu o hang 2
                                                        listHang1.pop(id+1 - count_)
                                                        count_ = count_ + 1

                                            # print('aaaaaaaaaaaa2')
                                            # print(listHang1)
                                            # print()
                                            # print('aaaaaaaaaaaa2')
                                            # kiểm tra list hàng 1 bị bỏ sót ký tự trong trường hợp xe máy bị nghiêng quá
                                            if (2*abs(y_max - y_min) > 3*trungvi_h):
                                                val_xmax0 = mangXuLy_sort_x[len(mangXuLy_sort_x) - 1]
                                                val_xmax1 = mangXuLy_sort_x[len(mangXuLy_sort_x) - 2]
                                                val_xmax2 = mangXuLy_sort_x[len(mangXuLy_sort_x) - 3]

                                                val_x0 = mangXuLy_sort_x[0]
                                                val_x1 = mangXuLy_sort_x[1]
                                                val_x2 = mangXuLy_sort_x[2]

                                                val_ymax = mangXuLy_sort_y[len(mangXuLy_sort_y) - 1]
                                                # truong hop nghieng ve ben trai
                                                # print(str(val_ymax))
                                                if (val_ymax == val_x0) or (val_ymax == val_x1) or (val_ymax == val_x2):
                                                    # kiem tra index 1, 2, 3, 4
                                                    val_x0 = mangXuLy_sort_x[0]
                                                    val_x1 = mangXuLy_sort_x[1]
                                                    val_x2 = mangXuLy_sort_x[2]
                                                    val_x3 = mangXuLy_sort_x[3]
                                                    val_x4 = mangXuLy_sort_x[4]
                                                    if (val_ymax != val_x0) and (val_x0 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x0, 2])
                                                        # print(str(names_nd[mangXuLy[val_x0, 0]]) + "   " + str(4*delda_y) + "   " + str(3*trungvi_h))
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x0)
                                                    if (val_ymax != val_x1) and (val_x1 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x1, 2])
                                                        # print(str(names_nd[mangXuLy[val_x1, 0]]) + "   " + str(4*delda_y) + "   " + str(3*trungvi_h))
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x1)
                                                    if (val_ymax != val_x2) and (val_x2 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x2, 2])
                                                        # print(str(names_nd[mangXuLy[val_x2, 0]]) + "   " + str(4*delda_y) + "   " + str(3*trungvi_h))
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x2)
                                                    if (val_ymax != val_x3) and (val_x3 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x3, 2])
                                                        # print(str(names_nd[mangXuLy[val_x3, 0]]) + "   " + str(4*delda_y) + "   " + str(3*trungvi_h))
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x3)
                                                    if (val_ymax != val_x4) and (val_x4 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x4, 2])
                                                        # print(str(names_nd[mangXuLy[val_x4, 0]]) + "   " + str(4*delda_y) + "   " + str(3*trungvi_h))
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x4)

                                                elif (val_ymax == val_xmax0) or (val_ymax == val_xmax1) or (val_ymax == val_xmax2):
                                                    val_x0 = mangXuLy_sort_x[len(mangXuLy_sort_y) - 1]
                                                    val_x1 = mangXuLy_sort_x[len(mangXuLy_sort_y) - 2]
                                                    val_x2 = mangXuLy_sort_x[len(mangXuLy_sort_y) - 3]
                                                    val_x3 = mangXuLy_sort_x[len(mangXuLy_sort_y) - 4]
                                                    val_x4 = mangXuLy_sort_x[len(mangXuLy_sort_y) - 5]
                                                    if (val_ymax != val_x0) and (val_x0 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x0, 2])
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x0)
                                                    if (val_ymax != val_x1) and (val_x1 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x1, 2])
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x1)
                                                    if (val_ymax != val_x2) and (val_x2 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x2, 2])
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x2)
                                                    if (val_ymax != val_x3) and (val_x3 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x3, 2])
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x3)
                                                    if (val_ymax != val_x4) and (val_x4 not in listHang1):
                                                        delda_y = abs(y_max - mangXuLy[val_x4, 2])
                                                        if 4*delda_y >= 3*trungvi_h:
                                                            listHang1.append(val_x4)

                                            hang1 = ""
                                            mangHang1 = mangXuLy[0]

                                            hang2 = ""
                                            xmin_ymin_hang2 = []
                                            mangHang2 = mangXuLy[0]
                                            toado_x_kytu_truoc = 0
                                            giatri_kytu_truoc = 0

                                            # print(listHang1)
                                            for idx, val in enumerate(mangXuLy_sort_x):

                                                if val in listHang1:
                                                    # print("hang1 " + str(names_nd[mangXuLy[val, 0]]))
                                                    if len(hang1) == 0:
                                                        mangHang1 = mangXuLy[val]

                                                        hang1 += str(names_nd[mangXuLy[val, 0]])
                                                        toado_x_kytu_truoc = mangXuLy[val, 1]
                                                        giatri_kytu_truoc = mangXuLy[val, 0]
                                                    else:
                                                        # kiem tra trung khop voi ky tu truoc
                                                        mangHang1_ = mangXuLy[val]
                                                        phantram = phanTramGiaoNhau(
                                                            mangHang1, mangHang1_)

                                                        if phantram > 0.8:
                                                            if mangHang1[5] < mangHang1_[5]:
                                                                # doi lai gia tri
                                                                mangHang1 = mangHang1_
                                                                hang1 = hang1[0: len(
                                                                    hang1) - 1] + str(names_nd[mangHang1_[0]])
                                                        else:
                                                            # khong trung thi them moi
                                                            isThemMoi_kytu = True
                                                            # kiem tra ky tu bi long vao nhau

                                                            if (toado_x_kytu_truoc > 0) and (giatri_kytu_truoc != 1):
                                                                delta_x_lienTiep = mangXuLy[val,
                                                                                            1] - toado_x_kytu_truoc
                                                                if 3 * delta_x_lienTiep < trungvi_w:
                                                                    isThemMoi_kytu = False

                                                            if isThemMoi_kytu:
                                                                toado_x_kytu_truoc = mangXuLy[val, 1]
                                                                mangHang1 = mangXuLy[val]
                                                                # hang1 += str(names_nd[mangXuLy[val, 0]])
                                                                hang1 += str(names_nd[mangXuLy[val, 0]])

                                                else:
                                                    if len(hang2) == 0:
                                                        kytu = str(names_nd[mangXuLy[val, 0]])
                                                        # if kytu.isdigit():
                                                        mangHang2 = mangXuLy[val]
                                                        hang2 += str(names_nd[mangXuLy[val, 0]])
                                                        xmin_ymin_hang2.append([mangXuLy[val, 1], mangXuLy[val, 2]])
                                                    else:
                                                        # kiem tra trung khop voi ky tu truoc
                                                        mangHang2_ = mangXuLy[val]
                                                        phantram = phanTramGiaoNhau(
                                                            mangHang2, mangHang2_)

                                                        if phantram > 0.8:
                                                            if mangHang2[5] < mangHang2_[5]:
                                                                # doi lai gia tri
                                                                mangHang2 = mangHang2_
                                                                hang2 = hang2[0:len(
                                                                    hang2) - 1] + str(names_nd[mangHang2_[0]])
                                                        else:
                                                            # khong trung thi them moi
                                                            kytu = str(names_nd[mangXuLy[val, 0]])
                                                            if kytu == 'B' or kytu == 'E':
                                                                kytu = '8'
                                                            # if kytu.isdigit():
                                                            mangHang2 = mangXuLy[val]
                                                            hang2 += str(names_nd[mangXuLy[val, 0]])
                                                            xmin_ymin_hang2.append([mangXuLy[val, 1], mangXuLy[val, 2]])
                                            # print('bbbbbbbbbbb ' + hang1 + '    ' + hang2)
                                            if maubien == 2 and len(hang1) == 2 and len(hang2) == 4:
                                                # if maubien == 0:
                                                #     maubien = 2
                                                loaixe = 2
                                                kytu0 = hang1[0]
                                                kytu1 = hang1[1]
                                                if kytu0 == '0':
                                                    kytu0 = 'Q'
                                                elif kytu0 == '8':
                                                    kytu0 = 'B'
                                                elif kytu0 == '7':
                                                    kytu0 = 'V'
                                                if kytu1 == '0':
                                                    kytu1 = 'Q'
                                                elif kytu1 == '8':
                                                    kytu1 = 'B'
                                                elif kytu1 == '1':
                                                    kytu1 = 'T'
                                                # bien do
                                                bienso = kytu0 + kytu1 + '-' + \
                                                    hang2[0:2] + '-' + hang2[2:4]
                                            else:
                                                # sua chu B nham thanh so 8
                                                if (len(hang1) == 4) and ((hang1[2] == '8') or (hang1[2] == '3')):
                                                    kytu0 = hang1[0]
                                                    kytu1 = hang1[1]
                                                    kytu2 = hang1[2]
                                                    kytu3 = hang1[3]

                                                    if kytu2 == '8':
                                                        kytu2 = 'B'
                                                    elif kytu2 == '3':
                                                        kytu2 = 'E'
                                                    bienso = kytu0 + kytu1 + kytu2 + kytu3 + '-' + hang2
                                                else:

                                                    bienso = hang1 + "-" + hang2

                                            if (len(hang1) > 5) or (len(hang2) > 6):
                                                print('...........')
                                                if ('NN' in hang1):
                                                    bienso = hang1[2:] + '-NN-' + hang2
                                                elif (('N' in hang1) and ('N' in hang2)):
                                                    bienso = hang1[1:3] + '-' + hang1[3:] + '-NN-' + hang2[1:]
                                                # bienso = ''
                                            else:

                                                if (len(hang1) > 2) and (loaixe < 2):
                                                    if len(hang1) == 3:
                                                        loaixe = 2
                                                    else:
                                                        if ('NN' in hang1):
                                                            bienso = hang1[2:] + '-NN-' + hang2
                                                        elif (('N' in hang1) and ('N' in hang2)):
                                                            bienso = hang1[1:3] + '-' + hang1[3:] + '-NN-' + hang2[1:]
                                                        else:
                                                            biensoFull = hang1 + hang2
                                                            if (len(biensoFull) >= 7) and (('NN' in biensoFull) or ('NG' in biensoFull) or ('HG' in biensoFull) or ('CV' in biensoFull) or ('QT' in biensoFull)):
                                                                loaixe = 2
                                                                if ('NN' in biensoFull):
                                                                    # 29-NG-636-11   29-636-NG-11
                                                                    index_char = biensoFull.index('NN')
                                                                    if index_char == 2:
                                                                        bienso = biensoFull[0:2] + '-NN-' + biensoFull[4:7] + '-' + biensoFull[7:]
                                                                    elif index_char == 5:
                                                                        bienso = biensoFull[0:2] + '-' + biensoFull[2:5] + '-NN-' + biensoFull[7:]
                                                                    else:
                                                                        bienso = biensoFull
                                                                elif ('NG' in biensoFull):
                                                                    # 29-NG-636-11   29-636-NG-11
                                                                    index_char = biensoFull.index('NG')
                                                                    if index_char == 2:
                                                                        bienso = biensoFull[0:2] + '-NG-' + biensoFull[4:7] + '-' + biensoFull[7:]
                                                                    elif index_char == 5:
                                                                        bienso = biensoFull[0:2] + '-' + biensoFull[2:5] + '-NG-' + biensoFull[7:]
                                                                    else:
                                                                        bienso = biensoFull
                                                                elif ('HG' in biensoFull):
                                                                    # 29-NG-636-11   29-636-NG-11
                                                                    index_char = biensoFull.index('NG')
                                                                    if index_char == 2:
                                                                        bienso = biensoFull[0:2] + '-NG-' + biensoFull[4:7] + '-' + biensoFull[7:]
                                                                    elif index_char == 5:
                                                                        bienso = biensoFull[0:2] + '-' + biensoFull[2:5] + '-NG-' + biensoFull[7:]
                                                                    else:
                                                                        bienso = biensoFull
                                                                elif ('CV' in biensoFull):
                                                                    # 29-NG-636-11   29-636-NG-11
                                                                    index_char = biensoFull.index('CV')
                                                                    if index_char == 2:
                                                                        bienso = biensoFull[0:2] + '-CV-' + biensoFull[4:7] + '-' + biensoFull[7:]
                                                                    elif index_char == 5:
                                                                        bienso = biensoFull[0:2] + '-' + biensoFull[2:5] + '-CV-' + biensoFull[7:]
                                                                    else:
                                                                        bienso = biensoFull
                                                                elif ('QT' in biensoFull):
                                                                    # 29-NG-636-11   29-636-NG-11
                                                                    index_char = biensoFull.index('QT')
                                                                    if index_char == 2:
                                                                        bienso = biensoFull[0:2] + '-QT-' + biensoFull[4:7] + '-' + biensoFull[7:]
                                                                    elif index_char == 5:
                                                                        bienso = biensoFull[0:2] + '-' + biensoFull[2:5] + '-QT-' + biensoFull[7:]
                                                                    else:
                                                                        bienso = biensoFull
                                                            elif ('LD' in hang1) or ('DA' in hang1) or ('KT' in hang1) or ('LA' in hang1)  or ('LB' in hang1) or ('CD' in hang1) or ('MK' in hang1):
                                                                loaixe = 2
                                                            elif 'MD' in hang1:
                                                                loaixe = -1
                                                            elif len(hang1) > 3:
                                                                kytu3 = hang1[3]
                                                                if kytu3.isdigit() == False:
                                                                    # xe gan may
                                                                    loaixe = 0
                                                # 2 ky tu dau khong phai la so, truong hop khong phai bien do thi bo qua
                                                # if bienso[0:2].isdigit() == False:
                                                #     bienso = ''
                                                # them dau - phan loai xe may
                                                if (len(bienso) >= 6) and (loaixe <= 1):
                                                    # print('aaaaaaaaaaaaaaaaaaaaa')
                                                    # kiem tra format bien 3 so
                                                    if (len(hang1) == 2) and (len(hang2) == 5) and (hang2[3:4].isdigit() == False):
                                                        bienso = hang1 + '-' + hang2[0:3] + '-' + hang2[3:]
                                                    else:
                                                        bienso = bienso[0:2] + '-' + bienso[2:]
                                                    # print('bbbbbbbbbbbbbbbb')
                                            # print(bienso)


                                            if len(bienso) > 0 and len(xmin_ymin_hang2) > 3:
                                                x_0 = xmin_ymin_hang2[0][0]
                                                y_0 = xmin_ymin_hang2[0][1]
                                                x_1 = xmin_ymin_hang2[1][0]
                                                y_1 = xmin_ymin_hang2[1][1]
                                                x_2 = xmin_ymin_hang2[2][0]
                                                y_2 = xmin_ymin_hang2[2][1]

                                                # radians_01 = math.atan((y_1 - y_0) / (x_1 - x_0));
                                                # angle_01 = radians_01 * (180 / math.pi);
                                                radians_12 = math.atan((y_2 - y_1) / (x_2 - x_1));
                                                angle_12 = radians_12 * (180 / math.pi);
                                                if angle_12 != 0:
                                                    gocnghieng = -angle_12

                                            if '--' in bienso:
                                                bienso = bienso.replace("--", "")
                                            elif (len(bienso) < 8) and ('-' in bienso):
                                                bienso = bienso.replace("-", "")

                            except Exception as e:
                                print('loi nhan dang bien so ' + str(e))
                                
                    delta_time = time.time() - toc
                    # print("time sap xep " + str(delta_time))
                traloi = {
                    "bienso": bienso,
                    "loaixe": loaixe,
                    "maubien": maubien,
                    "gocnghieng": gocnghieng,
                    "rec_x": int(recs_Crop[i][0]),
                    "rec_y": int(recs_Crop[i][1]),
                    "rec_w": int(recs_Crop[i][2] - recs_Crop[i][0]),
                    "rec_h": int(recs_Crop[i][3] - recs_Crop[i][1]),
                    "confident": confident_bs
                }
                list_traloi.append(traloi)
            break

    except Exception as e:
        print('loi ndbs ' + str(e))

    return list_traloi






def run(file_path:str):
                    
    queue_plate = {}
    counting_vehicle = {}
    check_counting = {}
    area_checked = {}
    object_tracked = {}
    correct_object = {}
    
    tracker = [OCSORT(asso_func='giou'), OCSORT(asso_func='giou'), OCSORT(asso_func='giou'), OCSORT(asso_func='giou')]
    check_bs = set()
    # color = (0, 0, 255)  # BGR  
    thickness = 3
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3        
    counting_lost = {}

    device = 0

    

    weights = "weight/phanloaixe_22072024.pt"

    weights_nd = "weight/best_bienso_31102024.pt"

    # Load model
    device = select_device(device)
    
    dnn = False
    data = 'data/coco128.yaml'
    half = True
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model_nd = DetectMultiBackend(weights_nd, device=device, dnn=dnn, data=data, fp16=half)
    color = (80,127,255) #màu tomato

    stride, names, pt = model.stride, model.names, model.pt
    stride_nd, names_nd, pt_nd = model_nd.stride, model_nd.names, model_nd.pt


    for key in names:
        if names[key].lower() != "plate":
            counting_vehicle[names[key]] = 0
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    imgsz_nd = (320, 320)
    
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    resultsBS_moto = [[], [], [], []]
    resultsBS_car = [[], [], [], []]
    resultsBS_bus_truck = [[], [], [], []]
    list_BienSo_LuuMois = []

    assigned_plates_by_tracker_id = OrderedDict()
    motor_plate = OrderedDict()
    plate = {}

    if file_path[0].endswith(".mp4"):
        cap = cv2.VideoCapture(file_path[0])
        while True:
            ret, im0 = cap.read()
            if not ret:
                break
            im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)

            used_plates = set()  # Biển số đã được sử dụng
       
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            pred = model(im)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)

            min_box_area = 10


            # ketqualuufiles_anhNguoiCrop = []
            # if len(pred) > 0:
            if len(list_BienSo_LuuMois) > 1000:
                del list_BienSo_LuuMois[i][0:50]

            for i, det in enumerate(pred):  # per image 
                # xóa tràn trong danh sách
                if(len(resultsBS_moto[i]) >= 1500):
                    del resultsBS_moto[i][0:50]

                if(len(resultsBS_car[i]) >= 1500):
                    del resultsBS_car[i][0:50]

                if(len(resultsBS_bus_truck[i]) >= 1500):
                    del resultsBS_bus_truck[i][0:50]            

           
        
    
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy()
                annotator = Annotator(im0, line_width=3, example=str(names))
                height0 = im0.shape[0]
                width0 = im0.shape[1]
                ts = 'tracker'
                if len(det):
                    box_gh_x0 = 20
                    box_gh_y0 = height0*0.1
                    box_gh_x1 = width0 - 20
                    box_gh_y1 = height0 - 20

                    # chứa danh sách rec biển số trong vùng giới hạn đọc biển
                    recs_Crop = []  
                    anhBienSo_Crop = []
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
                    det_goc_numpy = det.cpu().detach().numpy()
                    xyxys_goc = det_goc_numpy[:, 0:4].astype('int')
                    
                    xyxy_goc_moto = []
                    # for xyxy in xyxys_goc:
                    #     annotator.box_label(xyxy, '', color=colors(6, True))
                    clss_goc = det_goc_numpy[:, 5]
                    # tìm list rec biển số trong vung gioi han
                    for index, new_val_cls in enumerate(clss_goc):

                        c = int(new_val_cls)  # integer class
                        if c == 3:
                            xyxy_goc_moto.append(xyxys_goc[index])
                        # bien so
                        if c == 0:
                            x_min_crop_int = xyxys_goc[index][0]
                            y_min_crop_int = xyxys_goc[index][1]
                            x_max_crop_int = xyxys_goc[index][2]
                            y_max_crop_int = xyxys_goc[index][3]


                            # box_bs = box(int(x_min), int(y_min), int(x_max), int(y_max))
                            if (x_min_crop_int > box_gh_x0) and (x_max_crop_int < box_gh_x1)and (y_min_crop_int > box_gh_y0) and (y_max_crop_int < box_gh_y1):
                                width_recBS = x_max_crop_int - x_min_crop_int
                                height_recBS = y_max_crop_int - y_min_crop_int
                                if(height_recBS < 1.2*width_recBS):
                                    # recs_Crop.append([x_min_crop_int, y_min_crop_int, x_max_crop_int, y_max_crop_int])
                                    recBS = [x_min_crop_int, y_min_crop_int, x_max_crop_int, y_max_crop_int]
                                    # bienso_crop = im0[recBS[1]:recBS[3], recBS[0]:recBS[2]]                     
                                    # recs_Crop.append([x_min_crop_int, y_min_crop_int, x_max_crop_int, y_max_crop_int])
                                    # anhBienSo_Crop.append(bienso_crop)
                                    recs_Crop.append(recBS)
                    # print(det_goc_numpy )
                    ts = tracker[i].update(det_goc_numpy, im0) # --> (x, y, x, y, id, conf, cls)
                    try:
                        xyxys = ts[:, 0:4] # float64 to int
                        ids = ts[:, 4].astype('int') # float64 to int
                        confs = ts[:, 5]
                        clss = ts[:, 6]  

                        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                            if id not in assigned_plates_by_tracker_id and len(assigned_plates_by_tracker_id) > 150:
                                assigned_plates_by_tracker_id.popitem(last=False)
                            if int(cls) != 0:
                                if int(cls) == 3:
                                    toado_x1 = xyxy[0]
                                    toado_y1 = xyxy[1]
                                    toado_x2 = xyxy[2]
                                    toado_y2 = xyxy[3]
                                    toado_w = toado_x2 - toado_x1
                                    toado_h = toado_y2 - toado_y1
                                    toado_x2 = toado_x1 + toado_w/3
                                    toado_y2 = toado_y1 + toado_h/3     
                                    xyxy = [toado_x1, toado_y1, toado_x2, toado_y2]
                                x_center = (xyxy[0] + xyxy[2])/2
                                y_center = (xyxy[1] + xyxy[3])/2
                                if id not in correct_object:
                                    correct_object[id] = 1
                                else:
                                    correct_object[id] += 1

                                object_tracked[id] = (x_center, y_center) 
                                counting_lost[id] = 0

                        # check nhan dang bien so voi xe may truoc
                        if ts.shape[0] != 0:
                            listBienSoChoNhanDang = []
                            listBienSoChoNhanDang_Bitmap_only = []
                            listBienSoChoNhanDang_RecBS_only = []                    
                                
                            # xử lý với xe moto trước
                            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                                c = int(cls)
                                if c == 3:  
                                    isThemMoi = True
                                    ketquabienso = ''
                                    recBS_tuongung = ''
                                    # rec moto về đúng kích cỡ (vì đưa vào track là width và height motor đã được nhân lên 3 lần)
                                    toado_x1 = xyxy[0]
                                    toado_y1 = xyxy[1]
                                    toado_x2 = xyxy[2]
                                    toado_y2 = xyxy[3]
                                    toado_w = toado_x2 - toado_x1
                                    toado_h = toado_y2 - toado_y1
                                    toado_x2 = toado_x1 + toado_w/3
                                    toado_y2 = toado_y1 + toado_h/3     
                                    xyxy = [toado_x1, toado_y1, toado_x2, toado_y2]
                                    
                                    
                                    # tim xyxy trùng khớp với tọa độ hiện tại
                                    index_trungkhop = -1
                                    for index, xyxy_ in enumerate(xyxy_goc_moto):
                                        if xyxy_[0] == xyxy[0] and xyxy_[1] == xyxy[1]:
                                            xyxy = xyxy_
                                            index_trungkhop = index
                                            break
                                    if index_trungkhop > -1:
                                        del xyxy_goc_moto[index_trungkhop]
                                        
                                    limit_width_moto_check = 0


                                    isVeThongTin = True
                                    for index, resultBS in enumerate(resultsBS_moto[i]):
                                        recBS_tuongung = resultBS[4]                       
                                        if resultBS[0] == id:
                                            isThemMoi = False
                                            if (len(resultBS[1]) == 0 or resultBS[2] == False) and resultBS[3] <= 15:
                                                # doc bien so moto                                                                                
                                                # nhận dạng biển số
                                                toado_x1_check = toado_x1 - 50
                                                toado_y1_check = toado_y1
                                                toado_x2_check = toado_x2 + 50
                                                toado_y2_check = toado_y2 + 50
                                                if toado_x2 - toado_x1 >= limit_width_moto_check and toado_y2 - toado_y1 >= 0:                                                
                                                    if len(recs_Crop) > 0:
                                                        recBS_remove = 'null'
                                                        for recBS in recs_Crop:
                                                            if (recBS[0] >= toado_x1_check and recBS[1] >= toado_y1_check and recBS[2] <= toado_x2_check and recBS[3] <= toado_y2_check):                                        
                                                                recBS_remove = recBS                                                  
                                                                # nhận dạng biển số luôn
                                                                
                                                                bienso_crop = im0[recBS[1]:recBS[3], recBS[0]:recBS[2]]
                                                                listBienSoChoNhanDang.append([c, index, xyxy, id])
                                                                listBienSoChoNhanDang_Bitmap_only.append(bienso_crop)
                                                                listBienSoChoNhanDang_RecBS_only.append(recBS)
                                                                isVeThongTin = False
                                                                plate[id] = bienso_crop
                                                                # ketqua, ketquabiensofull = checkBienSo(bienso_crop, model_nd, names_nd, device, recBS[0], recBS[1], recBS[2] - recBS[0], recBS[3] - recBS[1], conf)
                                                            
                                                                # ketquabienso = ketqua                                           
                                                                # resultBS[1] = ketqua
                                                                # resultBS[2] = checkPlateNumberFormat_VN(ketqua)
                                                                # resultBS[3] += 1
                                                                break
                                                        if recBS_remove != 'null':
                                                            recs_Crop.remove(recBS_remove)

                                                # label = 'id: ' + str(id) + ' bs ' + str(ketqua_bienso)
                                                # annotator.box_label(xyxy, label, color=colors(c, True))
                                            else:
                                                ketquabienso = resultBS[1]

                                            break
                                    # print(isThemMoi)
                                    # xử lý trường hợp nhận nhầm biển số thành loại xe
                                    
                                            
                                    if toado_x2 - toado_x1 >= limit_width_moto_check and toado_y2 - toado_y1 >= 0:
                                        if isThemMoi:
                                            resultsBS_moto[i].append([id, '', False, 0, None])
                
                                        if isVeThongTin:
                                            if ketquabienso not in used_plates:
                                                label = str(ketquabienso)
                                                used_plates.add(ketquabienso)
                                            motor_plate[str(ketquabienso)] = True
                                            
                                            label = 'xe may ' + label

                                            
                                            if len(ketquabienso) >= 4:
                                                if recBS_tuongung is None:   
                                                    

                                                        
                                                    if id in plate and id not in check_bs and len(str(ketquabienso)) >= 6:
                                                        if len(queue_plate) >= 5:
                                                            pop(queue_plate)
                                                        queue_plate[str(ketquabienso)] = plate[id]
                                                        check_bs.add(id)
                                                        motor_plate[str(ketquabienso)] = True
                                                
                                                    annotator.box_label(xyxy, label, color)
                                                else:
                                                    width_xyxy = xyxy[2] - xyxy[0]
                                                    height_xyxy = xyxy[3] - xyxy[1]

                                                    if isinstance(recBS_tuongung, dict) and "rec_w" in recBS_tuongung:
                                                        if width_xyxy > 1.3*recBS_tuongung['rec_w'] and height_xyxy > 1.5*recBS_tuongung['rec_h']:

                                                            if id in plate and id not in check_bs and len(str(ketquabienso)) >= 6:
                                                                if len(queue_plate) >= 5:
                                                                    pop(queue_plate)
                                                                queue_plate[str(ketquabienso)] = plate[id]
                                                                check_bs.add(id)
                                                                motor_plate[str(ketquabienso)] = True
                                                            annotator.box_label(xyxy, label, color)


                            # xử lý với xe car
                            # print(motor_plate)
                            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                                c = int(cls)                  
                                if  c == 1:   
                                    isThemMoi = True
                                    ketquabienso = ''   
                                    recBS_tuongung = ''                      
                                    toado_x1 = xyxy[0]
                                    toado_y1 = xyxy[1]
                                    toado_x2 = xyxy[2]
                                    toado_y2 = xyxy[3]
                                    
                                    # toado_w = toado_x2 - toado_x1
                                    # toado_h = toado_y2 - toado_y1
                                    # toado_x2 = toado_x1 + toado_w/1.5
                                    # toado_y2 = toado_y1 + toado_h/1.5
                                    # xyxy = [toado_x1, toado_y1, toado_x2, toado_y2]
                                    limit_width_moto_check = 0
                                    isVeThongTin = True                                    
                                    for index, resultBS in enumerate(resultsBS_car[i]):
                                        recBS_tuongung = resultBS[4]                     
                                        if resultBS[0] == id:
                                            isThemMoi = False
                                            if (len(resultBS[1]) == 0 or resultBS[2] == False) and resultBS[3] <= 3:
                                                # nhận dạng biển số
                                                toado_x1_check = xyxy[0]
                                                toado_y1_check = xyxy[1]
                                                toado_x2_check = xyxy[2] 
                                                toado_y2_check = xyxy[3] + 50
                                                if toado_x2 - toado_x1 > limit_width_moto_check and toado_y2 - toado_y1 > 0:                                       
                                                    if len(recs_Crop) > 0:
                                                        recBS_remove = 'null'
                                                        for recBS in recs_Crop:
                                                            if (recBS[0] >= toado_x1_check and recBS[1] >= toado_y1_check and recBS[2] <= toado_x2_check and recBS[3] <= toado_y2_check):                                        
                                                                recBS_remove = recBS                                                           
                                                                # nhận dạng biển số luôn
                                                                bienso_crop = im0[recBS[1]:recBS[3], recBS[0]:recBS[2]]
                                                                listBienSoChoNhanDang.append([c, index, xyxy, id])
                                                                listBienSoChoNhanDang_Bitmap_only.append(bienso_crop)
                                                                listBienSoChoNhanDang_RecBS_only.append(recBS)
                                                                isVeThongTin = False
                                                                plate[id] = bienso_crop
                                                                # ketqua, ketquabiensofull = checkBienSo(bienso_crop, model_nd, names_nd, device, recBS[0], recBS[1], recBS[2] - recBS[0], recBS[3] - recBS[1], conf)
                                                                
                                                                # ketquabienso = ketqua                                           
                                                                # resultBS[1] = ketqua
                                                                # resultBS[2] = checkPlateNumberFormat_VN(ketqua)
                                                                # resultBS[3] += 1
                                                                break
                                                        if recBS_remove != 'null':
                                                            recs_Crop.remove(recBS_remove)
                                            else:
                                                ketquabienso = resultBS[1]             

                                            break
                                    # print(isThemMoi)
                                    # xử lý trường hợp nhận nhầm biển số thành loại xe
                                    
                                    if toado_x2 - toado_x1 > limit_width_moto_check and toado_y2 - toado_y1 > 0: 
                                        if isThemMoi:
                                            resultsBS_car[i].append([id, '', False, 0, None])                            
                                        if isVeThongTin:
                                            # label = str(id) + ' xe con ' + str(ketquabienso)
                                            if ketquabienso not in used_plates:
                                                used_plates.add(ketquabienso)
                                                label = str(ketquabienso)
                                            else:
                                                label = ""
                                            
                                            label = 'xe con ' + label
                                            if len(ketquabienso) >= 4 and ketquabienso not in motor_plate:
                                                
                                                if id in plate and id not in check_bs and len(str(ketquabienso)) >= 6:
                                                    if len(queue_plate) >= 5:
                                                        pop(queue_plate)
                                                    queue_plate[str(ketquabienso)] = plate[id]
                                                    check_bs.add(id)

                                            annotator.box_label(xyxy, label, color)
                            
                            # xử lý với xe truck, bus
                            for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                                c = int(cls)                  
                                if  c == 2 or c == 4:   
                                    isThemMoi = True
                                    ketquabienso = ''
                                    recBS_tuongung = ''
                                    toado_x1 = xyxy[0]
                                    toado_y1 = xyxy[1]
                                    toado_x2 = xyxy[2]
                                    toado_y2 = xyxy[3]
                                    isVeThongTin = True
                                    limit_width_moto_check = 0


                                    for index, resultBS in enumerate(resultsBS_bus_truck[i]):
                                        recBS_tuongung = resultBS[4]                          
                                        if resultBS[0] == id:
                                            isThemMoi = False
                                            if (len(resultBS[1]) == 0 or resultBS[2] == False) and resultBS[3] <= 15:
                                                # nhận dạng biển số
                                                toado_x1_check = xyxy[0]
                                                toado_y1_check = xyxy[1]
                                                toado_x2_check = xyxy[2] 
                                                toado_y2_check = xyxy[3] + 50
                                                if toado_x2 - toado_x1 > limit_width_moto_check and toado_y2 - toado_y1 > 0:                                       
                                                    if len(recs_Crop) > 0:
                                                        recBS_remove = 'null'
                                                        for recBS in recs_Crop:
                                                            if (recBS[0] >= toado_x1_check and recBS[1] >= toado_y1_check and recBS[2] <= toado_x2_check and recBS[3] <= toado_y2_check):                                        
                                                                recBS_remove = recBS
                                                                # nhận dạng biển số luôn
                                                                bienso_crop = im0[recBS[1]:recBS[3], recBS[0]:recBS[2]]
                                                                listBienSoChoNhanDang.append([c, index, xyxy, id])
                                                                listBienSoChoNhanDang_Bitmap_only.append(bienso_crop)
                                                                listBienSoChoNhanDang_RecBS_only.append(recBS)
                                                                isVeThongTin = False
                                                                plate[id] = bienso_crop
                                                                # ketqua, ketquabiensofull = checkBienSo(bienso_crop, model_nd, names_nd, device, recBS[0], recBS[1], recBS[2] - recBS[0], recBS[3] - recBS[1], conf)
                                                        
                                                                # ketquabienso = ketqua                                           
                                                                # resultBS[1] = ketqua
                                                                # resultBS[2] = checkPlateNumberFormat_VN(ketqua)
                                                                # resultBS[3] += 1
                                                                break
                                                        if recBS_remove != 'null':
                                                            recs_Crop.remove(recBS_remove)
                                            else:
                                                ketquabienso = resultBS[1]

                                            break
                                    # print(isThemMoi)
                                    # xử lý trường hợp nhận nhầm biển số thành loại xe
                                    

                                    if toado_x2 - toado_x1 > limit_width_moto_check and toado_y2 - toado_y1 > 0: 
                                        if isThemMoi:
                                            resultsBS_bus_truck[i].append([id, '', False, 0, None])
                                        if isVeThongTin:
                                            if c == 2:
                                                # label = str(id) + ' xe tai ' + str(ketquabienso)
                                                if ketquabienso not in used_plates:
                                                    # print(str(ketquabienso))
                                                    label = str(ketquabienso)
                                                    used_plates.add(ketquabienso)
                                                else:
                                                    label = ""
                                                label = 'xe tai ' + label
                                            elif c == 4:
                                                # label = str(id) + ' xe buyt/khach ' + str(ketquabienso)  
                                                if ketquabienso not in used_plates:
                                                    # print("ketquabienso", ketquabienso)
                                                    label = str(ketquabienso)
                                                    used_plates.add(ketquabienso)
                                                else:
                                                    label = ""
                                               
                                                label = 'xe buyt/khach ' + label
    
                                            if len(ketquabienso) >= 4 :  
                                                # idx = np.where(ids == id)[0]
                                                # c_obj = clss[idx]
                                                
                                                # if int(c_obj[0]) != 0:
                                                
                                                if id in plate and id not in check_bs and len(str(ketquabienso)) >= 6:
                                                    if len(queue_plate) >= 5:
                                                        pop(queue_plate)
                                                    queue_plate[str(ketquabienso)] = plate[id]
                                                    check_bs.add(id)
                                            annotator.box_label(xyxy, label, color)

                        if len(listBienSoChoNhanDang_Bitmap_only) > 0:
                            # nhan dang cung luc nhiều ảnh
                            list_traloi = checkBienSo2(listBienSoChoNhanDang_Bitmap_only, model_nd, names_nd, device, listBienSoChoNhanDang_RecBS_only, 0)   
                        
                            for index, traloi in enumerate(list_traloi):    
                                # traloi = {
                                #     "bienso": bienso,
                                #     "loaixe": loaixe,
                                #     "maubien": maubien,
                                #     "gocnghieng": gocnghieng,
                                #     "rec_x": recs_Crop[i][0],
                                #     "rec_y": recs_Crop[i][1],
                                #     "rec_w": recs_Crop[i][2] - recs_Crop[i][0],
                                #     "rec_h": recs_Crop[i][3] - recs_Crop[i][1],
                                #     "confident": confident_bs
                                # }

                                bienso_text = traloi['bienso']   
                                if(bienso_text != ''):
                                    # xyxy_traloi = [traloi['rec_x'], traloi['rec_y'], traloi['rec_x'] + traloi['rec_w'], traloi['rec_y'] + traloi['rec_h']]
                                    # annotator.box_label(xyxy_traloi, traloi['bienso'], color=colors(0, True))
                                    if listBienSoChoNhanDang[index][0] == 3:
                                        # xe moto   
                                        resultsBS_moto[i][listBienSoChoNhanDang[index][1]][1] = bienso_text
                                        resultsBS_moto[i][listBienSoChoNhanDang[index][1]][2] = checkPlateNumberFormat_VN(bienso_text)
                                        resultsBS_moto[i][listBienSoChoNhanDang[index][1]][3] += 1
                                        resultsBS_moto[i][listBienSoChoNhanDang[index][1]][4] = traloi
                                    elif listBienSoChoNhanDang[index][0] == 1:
                                        # xe car 
                                        resultsBS_car[i][listBienSoChoNhanDang[index][1]][1] = bienso_text
                                        resultsBS_car[i][listBienSoChoNhanDang[index][1]][2] = checkPlateNumberFormat_VN(bienso_text)
                                        resultsBS_car[i][listBienSoChoNhanDang[index][1]][3] += 1
                                        resultsBS_car[i][listBienSoChoNhanDang[index][1]][4] = traloi
                                    elif listBienSoChoNhanDang[index][0] == 2 or listBienSoChoNhanDang[index][0] == 4:
                                        # xe bus/truck 
                                        resultsBS_bus_truck[i][listBienSoChoNhanDang[index][1]][1] = bienso_text
                                        resultsBS_bus_truck[i][listBienSoChoNhanDang[index][1]][2] = checkPlateNumberFormat_VN(bienso_text)
                                        resultsBS_bus_truck[i][listBienSoChoNhanDang[index][1]][3] += 1
                                        resultsBS_bus_truck[i][listBienSoChoNhanDang[index][1]][4] = traloi
                                else:
                                    if listBienSoChoNhanDang[index][0] == 3:
                                        # xe moto   
                                        bienso_text = resultsBS_moto[i][listBienSoChoNhanDang[index][1]][1]
                                    elif listBienSoChoNhanDang[index][0] == 1:
                                        # xe car 
                                        bienso_text = resultsBS_car[i][listBienSoChoNhanDang[index][1]][1]
                                    elif listBienSoChoNhanDang[index][0] == 2 or listBienSoChoNhanDang[index][0] == 4:
                                        # xe bus/truck 
                                        bienso_text = resultsBS_bus_truck[i][listBienSoChoNhanDang[index][1]][1]

                                if len(bienso_text) >= 4:
                                    if c == 2:
                                        xyxy = listBienSoChoNhanDang[index][2]
                                        id = listBienSoChoNhanDang[index][3]
                                        recBS = listBienSoChoNhanDang_RecBS_only[index]
                                        if str(bienso_text) not in used_plates:
                                            label = bienso_text
                                            used_plates.add(bienso_text)
                                        else:
                                            bienso_text = ""
                                            label = ""
                                        label = 'xe tai ' + label
                                        
                                        if len(bienso_text) >= 4 and bienso_text not in motor_plate:
                                            
                                            if id in plate and id not in check_bs and len(str(bienso_text)) >= 6:
                                                if len(queue_plate) >= 5:
                                                    pop(queue_plate)
                                                queue_plate[str(ketquabienso)] = plate[id]
                                                check_bs.add(id)
                                            
                                        annotator.box_label(xyxy, label, color)
                                    elif c == 4:
                                        xyxy = listBienSoChoNhanDang[index][2]                            
                                        id = listBienSoChoNhanDang[index][3]
                                        recBS = listBienSoChoNhanDang_RecBS_only[index]
                                        # label = str(id) + ' xe buyt/khach ' + str(bienso_text)   
                                        
                                        if bienso_text not in used_plates:
                                            label = str(bienso_text)
                                            used_plates.add(bienso_text)
                                        else:
                                            bienso_text = ""
                                            label = ""
                                        label = 'xe buyt/khach ' + label
                                        if len(bienso_text) >= 4 and bienso_text not in motor_plate:
                                            
                                            if id in plate and id not in check_bs and len(str(bienso_text)) >= 6:
                                                if len(queue_plate) >= 5:
                                                    pop(queue_plate)
                                                queue_plate[str(bienso_text)] = plate[id]
                                                check_bs.add(id)
                                        annotator.box_label(xyxy, label, color)
                                    elif c == 1:
                                        xyxy = listBienSoChoNhanDang[index][2]                                
                                        id = listBienSoChoNhanDang[index][3]
                                        recBS = listBienSoChoNhanDang_RecBS_only[index]
                                        # label = str(id) + ' xe con ' + str(bienso_text)
                                        
                                        if bienso_text not in used_plates:
                                            label = str(bienso_text)
                                            used_plates.add(bienso_text)
                                        else:
                                            bienso_text = ""
                                            label = ""
                                        label = 'xe con ' + label
                                        
                                        if len(bienso_text) >= 4 and bienso_text not in motor_plate:
                                            
                                            if id in plate and id not in check_bs and len(str(bienso_text)) >= 6:
                                                if len(queue_plate) >= 5:
                                                    pop(queue_plate)
                                                queue_plate[str(bienso_text)] = plate[id]
                                                check_bs.add(id)
                                        annotator.box_label(xyxy, label, color)
                                    elif c == 3:
                                        xyxy = listBienSoChoNhanDang[index][2] 
                                        id = listBienSoChoNhanDang[index][3]
                                        recBS = listBienSoChoNhanDang_RecBS_only[index]
                                        
                                        # label = str(id) + ' xe may ' + str(bienso_text)
                                        if bienso_text not in used_plates:
                                            label = str(bienso_text)
                                            used_plates.add(bienso_text)
                                        else:
                                            bienso_text = ""
                                            label = ""

                                        label = 'xe may ' + label
                                        
                                        
                                        if len(bienso_text) >= 4:
                                            idx = np.where(ids == id)[0]
                                            c_obj = clss[idx]
                                            
                                            # if int(c_obj[0]) != 0:
                                            if id in plate and id not in check_bs and len(str(bienso_text)) >= 6:
                                                if len(queue_plate) >= 5:
                                                    pop(queue_plate)
                                                queue_plate[str(bienso_text)] = plate[id]
                                                check_bs.add(id)
                                                motor_plate[str(bienso_text)] = True
                                        
                                        annotator.box_label(xyxy, label, color)
                                    else:
                                        xyxy = listBienSoChoNhanDang[index][2] 
                                        id = listBienSoChoNhanDang[index][3]
                                        recBS = listBienSoChoNhanDang_RecBS_only[index]
                                        # label = str(id) + str(bienso_text)
                                        
                                        if bienso_text not in used_plates:
                                            label = str(bienso_text)
                                            used_plates.add(bienso_text)
                                        else:
                                            bienso_text = ""
                                            label = ""
                                       
                                        
                                        
                                       
                                        if len(bienso_text) >= 4 and bienso_text not in motor_plate:
                                            if id in plate and id not in check_bs and len(str(bienso_text)) >= 6:
                                                if len(queue_plate) >= 5:
                                                    pop(queue_plate)
                                                queue_plate[str(bienso_text)] = plate[id]
                                                check_bs.add(id)
                                                
                                        annotator.box_label(xyxy, label, ())
                                    


                    except Exception as e:
                        raise('xu ly tung frame ' + str(e))
                
                
            try:

                show_image = annotator.result()

                
                show_image = overlay_plates_on_frame(show_image, plates_dict=queue_plate, right_padding=config["rightPaddingShowPlate"], y_start_show_plate=config["yShowPlateStart"], y_magin = config["yShowMargin"], plate_size=config["PlateSize2Show"], transparency = config["transparencyPlateShow"])
                im_ = cv2.resize(show_image, (1280, 720))           

                cv2.imshow("frame2", im_)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                # raise(e)
                raise('loi hien thi ' + str(e))
    else:
        im0 = cv2.imread(file_path[0])
        im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        annotator = Annotator(im0, line_width=3, example=str(names))
        used_plates = set()  # Biển số đã được sử dụng
        queue_plate = {}
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
        listBienSoChoNhanDang_Bitmap_only = []
        listBienSoChoNhanDang_RecBS_only = []
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            clss = det[:, 5]
            det_goc_numpy = det.cpu().detach().numpy()
            xyxys_goc = det_goc_numpy[:, 0:4].astype('int')
            for xyxy, cls in zip(xyxys_goc, clss):
                if cls == 0:
                    x_min_crop_int = xyxy[0]
                    y_min_crop_int = xyxy[1]
                    x_max_crop_int = xyxy[2]
                    y_max_crop_int = xyxy[3]
                    img_bs = im0[y_min_crop_int:y_max_crop_int, x_min_crop_int:x_max_crop_int, :]
                    
                    listBienSoChoNhanDang_Bitmap_only.append(img_bs)
                    listBienSoChoNhanDang_RecBS_only.append(xyxy)
            list_traloi = checkBienSo2(listBienSoChoNhanDang_Bitmap_only, model_nd, names_nd, device, listBienSoChoNhanDang_RecBS_only, 0)
            for xyxy, cls in zip(xyxys_goc, clss):
                
                if cls != 0:
                    if cls == 3:
                        label = "xe may"
                    if cls == 2:
                        label = "xe tai"
                    if cls == 1:
                        label = "xe con"
                    if cls == 4:
                        label = "xe khach/bus"
                    for xyxy_, ketquabienso, BienSoChoNhanDang_Bitmap in zip(listBienSoChoNhanDang_RecBS_only, list_traloi, listBienSoChoNhanDang_Bitmap_only):
                        if is_nested(xyxy, xyxy_) and len(ketquabienso["bienso"]) >= 4 :
                            label += " " + ketquabienso["bienso"]
                            queue_plate[ketquabienso["bienso"]] = BienSoChoNhanDang_Bitmap

                    annotator.box_label(xyxy, label, color)
            show_image = annotator.result()
            show_image = overlay_plates_on_frame(show_image, plates_dict=queue_plate, right_padding=config["rightPaddingShowPlate"], y_start_show_plate=config["yShowPlateStart"], y_magin = config["yShowMargin"], plate_size=config["PlateSize2Show"], transparency = config["transparencyPlateShow"])
            im_ = cv2.resize(show_image, (1280, 720))
            cv2.imshow("frame2", im_)
            cv2.waitKey(0)


if __name__ == '__main__':
    while True:
        path = select_files()
        run(path)
        if 0xFF == ord('e'):
                break