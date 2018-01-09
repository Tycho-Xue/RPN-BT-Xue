import cv2
from keras_frcnn import data_generators

def plot_bbox(img_data,C,bbox):
    img_original=cv2.imread(img_data['filepath'])
    (width, height)=(img_data['width'], img_data['height'])
    (rows, cols, _) = img_original.shape
    path_old = img_data['filepath']
    assert cols == width
    assert rows == height
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    if bbox is not None:
        for i in range(bbox.shape[1]):
            x_c = bbox[0][i][0]
            y_c = bbox[0][i][1]
            w = bbox[0][i][2]
            h = bbox[0][i][3]
            x1_resize = x_c - w/2.0
            y1_resize = y_c - w/2.0
            x2_resize = x_c + h/2.0
            y2_resize = y_c + h/2.0
        
            x1_original = x1_resize * (float(width)/resized_width)
            x2_original = x2_resize * (float(width)/resized_width)
            y1_original = y1_resize * (float(height)/resized_height)
            y2_original = y2_resize * (float(height)/resized_height)

            x1 = int(round(x1_original*C.rpn_stride))
            x2 = int(round(x2_original*C.rpn_stride))
            y1 = int(round(y1_original*C.rpn_stride))
            y2 = int(round(y2_original*C.rpn_stride))
        
            cv2.rectangle(img_original,(x1,y1),(x2,y2),(255,0,0),2)

    img_ID_start = path_old.find('/JPEGImages/')
    img_ID = path_old[img_ID_start+12:]
    img_addr = '/home/xuele/rpn_bf/faster_rcnn/keras-frcnn/bbox_plot/'+img_ID
    cv2.imwrite(img_addr,img_original)





