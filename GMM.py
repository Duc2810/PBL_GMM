import numpy as np
import cv2 as cv
from object_tracker import EuclideanDistTracker

end = 0

#tao Tracker Object

f = 25
#cap = cv2.VideoCapture("test2.mp4")
w = int(1000/(f-1))
# khai bao object tracker va dem frame
tracker = EuclideanDistTracker()
frame_idx = 0
# khai bao ham gaussian distribution pdf
def gauss_dis_pdf(x, mean, sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

# Lay dau vao video de phat hien doi tuong
cap = cv.VideoCapture(r"C:\Users\trang\Downloads\video_pbl4\2.mp4")
#cap = cv.VideoCapture(r"C:\Users\trang\Downloads\finalllll.mp4")

_, frame = cap.read()

# lua chon khu vuc theo doi tren frame
roi = frame
#roi = frame[200:600, 300:1000]
# roi = frame[340: 720,500: 800]

# chuyen doi roi thanh anh xam
roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

# lay kich thuoc cua roi
row, col = roi.shape
               
# 3 mixtures of gaussian :

# khởi tạo mean,var,omega và omega theo sigma
mean = np.zeros([3, row, col], np.float64)
mean[:, :, :] = 0


mean_43 = np.zeros([3, row, col], np.float64)
mean_43[:, :, :] = 0

var = np.zeros([3, row, col], np.float64)
var[:, :, :] = 0


# Omega - trọng lượng một gaussian (tỉ lệ tồn tại trong khung hình)
omega = np.zeros([3, row, col], np.float64)
omega[:, :, :] = 0

# tỉ lệ Omega / sigma
r = np.zeros([3, row, col], np.float64)

# khởi tạo foreground và background
foreground = np.zeros([row, col], np.uint8)
background = np.zeros([row, col], np.uint8)

# khởi tạo tốc độ học tập alpha và ngưỡng T
alpha = 0.3
T = 0.5

# converting data type of integers 0 and 255 to uint8 type
#a = np.uint8([255]) # White frame
#b = np.uint8([0]) # Black frame


kernalOp = np.ones((1,1),np.uint8)
kernalCl = np.ones((13,13),np.uint8)
kernal_e = np.ones((7,7),np.uint8)
kernel = np.ones((5,5), np.uint8)



while cap.isOpened():
    _, frame = cap.read()
    # roi = frame[340: 720,500: 800]
    roi = frame
    #roi = frame[200:600, 300:1000]
    frame_idx += 1
    
    # Chuyển đổi roi thành ảnh xám để có thể áp dụng phân phối gaussian 1D
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)

    # phương sai var sẽ có giá trị âm sau một thời gian áp dụng hàm gauss_dis_pdf 
    # giá trị gần bằng 0 đến một số giá trị cao hơn theo tùy chọn của chúng
    var[0][np.where(var[0] < 1)] = 10
    var[1][np.where(var[1] < 1)] = 5
    var[2][np.where(var[2] < 1)] = 1


    # tính độ lệch chuẩn

    sigma1 = np.sqrt(var[0])
    sigma2 = np.sqrt(var[1])
    sigma3 = np.sqrt(var[2])

    # test dừng cập nhật các mô hình gauss ở frame thứ n trở đi:
    # if ((frame_idx >= 40 )) : 
    #     if ((frame_idx == 40 )):

    #             mean_43[0] = mean[0]
    #             mean_43[1] = mean[1]
    #             mean_43[2] = mean[2]

    #             value1_43 = 2.5 * sigma1
    #             value2_43 = 2.5 * sigma2
    #             value3_43 = 2.5 * sigma3 
    #     print(frame_idx)
    #     value1 = value1_43
    #     value2 = value2_43
    #     value3 = value3_43

    #     mean[0] = mean_43[0]
    #     mean[1] = mean_43[1]
    #     mean[2] = mean_43[2]

    # else :
    #     value1 = 2.5 * sigma1
    #     value2 = 2.5 * sigma2
    #     value3 = 2.5 * sigma3
#############################################

    #test chỉ cập nhật 1 lần mỗi 3 frame :

    # if (frame_idx % 3) == 0 :
        
    #     print(frame_idx)

    #     mean_43[0] = mean[0]
    #     mean_43[1] = mean[1]
    #     mean_43[2] = mean[2]

    #     value1_43 = 2.5 * sigma1
    #     value2_43 = 2.5 * sigma2
    #     value3_43 = 2.5 * sigma3
    # else :
    #     if (frame_idx == 1) :
    #         value1_43 = 2.5 * sigma1
    #         value2_43 = 2.5 * sigma2
    #         value3_43 = 2.5 * sigma3

    #     value1 = value1_43
    #     value2 = value2_43
    #     value3 = value3_43

    #     mean[0] = mean_43[0]
    #     mean[1] = mean_43[1]
    #     mean[2] = mean_43[2]

        
#############################################

    # tính các giá trị (X  -  mean thứ k)
    compare_val_1 = cv.absdiff(gray, mean[0])
    compare_val_2 = cv.absdiff(gray, mean[1])
    compare_val_3 = cv.absdiff(gray, mean[2])


    # ngưỡng phù hợp với gaussian thứ k (2.5*độ lệch chuẩn thứ k)

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    #xác định background thông qua ngưỡng T
    fore_index1 = np.where(omega[2] > T)
    fore_index2 = np.where(((omega[2]+omega[1]) > T) & (omega[2] < T))

    #Tìm các chỉ số của một giá trị pixel cụ thể phù hợp với ít nhất một trong các gaussian :
    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    temp = np.zeros([row, col])
    temp[fore_index1] = 1
    temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
    index3 = np.where(temp == 2)

    temp = np.zeros([row, col])
    temp[fore_index2] = 1
    index = np.where((compare_val_3 <= value3) | (compare_val_2 <= value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp == 2)

    match_index = np.zeros([row, col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

    # Cập nhật phương sai, giá trị trung bình và trọng lượng của các chỉ số phù hợp của cả ba gaussian
    # Gaussian1
    Fitness = alpha * gauss_dis_pdf(gray[gauss_fit_index1],
                                mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])


    constant = Fitness * ((gray[gauss_fit_index1] -
                      mean[0][gauss_fit_index1]) ** 2)


    mean[0][gauss_fit_index1] = (
        1 - Fitness) * mean[0][gauss_fit_index1] + Fitness * gray[gauss_fit_index1]


    var[0][gauss_fit_index1] = (1 - Fitness) * var[0][gauss_fit_index1] + constant


    omega[0][gauss_fit_index1] = (
        1 - alpha) * omega[0][gauss_fit_index1] + alpha


    omega[0][gauss_not_fit_index1] = (
        1 - alpha) * omega[0][gauss_not_fit_index1]

    # Gaussian2
    Fitness = alpha * gauss_dis_pdf(gray[gauss_fit_index2],
                                mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])


    constant = Fitness * ((gray[gauss_fit_index2] -
                      mean[1][gauss_fit_index2]) ** 2)


    mean[1][gauss_fit_index2] = (
        1 - Fitness) * mean[1][gauss_fit_index2] + Fitness * gray[gauss_fit_index2]


    var[1][gauss_fit_index2] = (
        1 - Fitness) * var[1][gauss_fit_index2] + Fitness * constant


    omega[1][gauss_fit_index2] = (
        1 - alpha) * omega[1][gauss_fit_index2] + alpha


    omega[1][gauss_not_fit_index2] = (
        1 - alpha) * omega[1][gauss_not_fit_index2]

    # Gaussian3
    Fitness = alpha * gauss_dis_pdf(gray[gauss_fit_index3],
                                mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])


    constant = Fitness * ((gray[gauss_fit_index3] -
                      mean[2][gauss_fit_index3]) ** 2)


    mean[2][gauss_fit_index3] = (
        1 - Fitness) * mean[2][gauss_fit_index3] + Fitness * gray[gauss_fit_index3]


    var[2][gauss_fit_index3] = (1 - Fitness) * var[2][gauss_fit_index3] + constant


    omega[2][gauss_fit_index3] = (
        1 - alpha) * omega[2][gauss_fit_index3] + alpha


    omega[2][gauss_not_fit_index3] = (
        1 - alpha) * omega[2][gauss_not_fit_index3]

    # Cập nhật gaussian ít có khả năng xảy ra nhất cho những giá trị pixel không khớp với bất kỳ gaussian nào
    mean[0][not_match_index] = gray[not_match_index]
    var[0][not_match_index] = 400
    omega[0][not_match_index] = 1

    # chuẩn hóa omega
    sum = np.sum(omega, axis=0)
    omega = omega/sum

    # Tìm tỷ lệ omega theo sigma để lấy nền và tiền cảnh
    r[0] = omega[0] / sigma1
    r[1] = omega[1] / sigma2
    r[2] = omega[2] / sigma3

    # tìm giá trị chỉ mục dựa trên r
    index = np.argsort(r, axis=0)

    #  sắp xếp mean,var and omega
    mean = np.take_along_axis(mean, index, axis=0)
    var = np.take_along_axis(var, index, axis=0)
    omega = np.take_along_axis(omega, index, axis=0)

    gray = gray.astype(np.uint8)

    # xác định background :
    background[index2] = gray[index2]
    background[index3] = gray[index3]

    # Object Detection
    mask = cv.subtract(gray, background)
    _, mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)
    
    mask1 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernalOp)
    mask11 = cv.dilate(mask1, kernel, iterations=1)
    mask2 = cv.morphologyEx(mask11, cv.MORPH_CLOSE, kernalCl)
    e_img = cv.erode(mask2, kernal_e)


    # Applying contour to mask
    contours, _ = cv.findContours(e_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    detection_dict = []

    for i in contours:
        # Tinh pham vi va loai bo nhieu
        area = cv.contourArea(i)
        # chon cac đoi tuong chuyen đong dua tren su thay đoi D Tich theo đuong vien
        if area > 900:

            x, y, w, h = cv.boundingRect(i)
            detection_dict.append([x, y, w, h])

    object_ids = tracker.update(detection_dict)
    for object_id in object_ids:
        x, y, w, h, id = object_id
        #so sanh toc do :
        if(tracker.getsp(id)<tracker.limit()):
            #toc do duoi gioi han :
            cv.putText(roi,str(id)+" "+str(tracker.getsp(id))+"km/h",(x,y-15), cv.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            #toc do vuot gioi han :
            cv.putText(roi,str(id)+" "+str(tracker.getsp(id))+"km/h",(x, y-15),cv.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)
        # luu hinh anh xe vuot toc do :
        s = tracker.getsp(id)
        if ( tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id)


    
    cv.line(roi, (0, 275), (540, 275), (255, 0, 255), 2)
    cv.line(roi, (0, 330), (540, 330), (0, 0, 255), 2)

    cv.line(roi, (0, 580), (540, 580), (0, 255, 255), 2)
    cv.line(roi, (0, 635), (540, 635), (0, 0, 255), 2)



    #cv.imshow('Object tracking', roi)
    
    cv.imshow('Mask',mask1)
    #cv.imshow("Frame", frame)
    
    key = cv.waitKey(50)
    # go 'space' de dung
    if key == 32:
        tracker.end()
        end = 1
        break

if(end!=1):
    tracker.end()

cap.release()
cv.destroyAllWindows()
