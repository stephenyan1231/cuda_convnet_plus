'''Copyright (c) 2014 Zhicheng Yan (zhicheng.yan@live.com)
'''
import os
import sys
import numpy as np
import numpy.linalg as LA
import time
import scipy.misc
import scipy.ndimage
from skimage import color
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplot
import multiprocessing as mtp
import exifread
import scipy.interpolate

import tifffile as tiff

class UtilImageError(Exception):
    pass


CENTRAL_PX_FEATURE_DIM = 3
CENTRAL_PX_COLOR_DIM = 3
CENTRAL_PX_MOMENTS_DIM = 3 * 3
CENTRAL_PX_GRAD_VECTOR_DIM = 3 * 2
CENTRAL_PX_BLOCK_CORRELATION_DIM = 3 * 4




''' undo gamma correction '''
def linearize_ProPhotoRGB(pp_rgb, reverse=False):
    if not reverse:
        gamma = 1.8
    else:
        gamma = 1.0/1.8
    pp_rgb = np.power(pp_rgb, gamma)
    return pp_rgb
'''
def linearize_SRGB(srgb):
    gamma = 2.2  # use approximation
    return np.power(srgb, gamma) 

def gamma_correction_SRGB(srgb):
    gamma = 2.2
    return np.power(srgb, 1.0 / gamma)
'''  

def XYZ_chromatic_adapt(xyz, src_white='D65', dest_white='D50'):
    if src_white == 'D65' and dest_white == 'D50':
        M = [[1.0478112, 0.0228866, -0.0501270], \
           [0.0295424, 0.9904844, -0.0170491], \
           [-0.0092345, 0.0150436, 0.7521316]]
    elif src_white == 'D50' and dest_white == 'D65':
        M = [[0.9555766, -0.0230393, 0.0631636], \
           [-0.0282895, 1.0099416, 0.0210077], \
           [0.0122982, -0.0204830, 1.3299098]]
    else:
        raise UtilCnnImageEnhanceError('invalid pair of source and destination white reference %s,%s')\
            % (src_white, dest_white)
    M = np.array(M)
    sp = xyz.shape
    assert sp[2] == 3
    xyz = np.transpose(np.dot(M, np.transpose(xyz.reshape((sp[0] * sp[1], 3)))))
    return xyz.reshape((sp[0], sp[1], 3))

'''
# white reference is 'D65'
def SRGB2XYZ(src, dir='forward'):
    if dir == 'forward':
        M = [[0.4124564, 0.3575761, 0.1804375], \
           [0.2126729, 0.7151522, 0.0721750], \
           [0.0193339, 0.1191920, 0.9503041]]
    elif dir == 'backward':
        M = [[3.2404542, -1.5371385, -0.4985314], \
           [-0.9692660, 1.8760108, 0.0415560], \
           [0.0556434, -0.2040259, 1.0572252]]        
    else:
        raise UtilImageError('unknown direction: %s' % dir)
    M = np.array(M) 
    sp = src.shape
    assert sp[2] == 3
    dest = np.transpose(np.dot(M, np.transpose(src.reshape((sp[0] * sp[1], 3)))))
    return dest.reshape((sp[0], sp[1], 3))
'''

# pp_rgb float in range [0,1]
# refernce white is D50
def ProPhotoRGB2XYZ(pp_rgb,reverse=False):
    if not reverse:
        M = [[0.7976749, 0.1351917, 0.0313534], \
           [0.2880402, 0.7118741, 0.0000857], \
           [0.0000000, 0.0000000, 0.8252100]]
    else:
        M = [[ 1.34594337, -0.25560752, -0.05111183],\
             [-0.54459882,  1.5081673,   0.02053511],\
             [ 0,          0,          1.21181275]]
    M = np.array(M)
    sp = pp_rgb.shape
    xyz = np.transpose(np.dot(M, np.transpose(pp_rgb.reshape((sp[0] * sp[1], sp[2])))))
    return xyz.reshape((sp[0], sp[1], 3))

''' normalize L channel so that minimum of L is 0 and maximum of L is 100 '''
def normalize_Lab_image(lab_image):
    h, w, ch = lab_image.shape[0], lab_image.shape[1], lab_image.shape[2]
    assert ch == 3
    lab_image = lab_image.reshape((h * w, ch))
    L_ch = lab_image[:, 0]
    L_min, L_max = np.min(L_ch), np.max(L_ch)
#     print 'before normalization L min %f,Lmax %f' % (L_min,L_max)
    scale = 100.0 / (L_max - L_min)
    lab_image[:, 0] = (lab_image[:, 0] - L_min) * scale
#     print 'after normalization L min %f,Lmax %f' %\
    (np.min(lab_image[:, 0]), np.max(lab_image[:, 0]))
    lab_image = lab_image.reshape((h, w, ch))

   

''' white reference 'D65' '''
def read_tiff_16bit_img_into_XYZ(tiff_fn, exposure=0):
    pp_rgb = tiff.imread(tiff_fn)
    pp_rgb = np.float64(pp_rgb) / (2 ** 16 - 1.0) 
    if not pp_rgb.shape[2] == 3:
        print 'pp_rgb shape',pp_rgb.shape
        raise UtilImageError('image channel number is not 3')
    pp_rgb = linearize_ProPhotoRGB(pp_rgb)
    pp_rgb *= np.power(2, exposure)
    xyz = ProPhotoRGB2XYZ(pp_rgb)
    xyz = XYZ_chromatic_adapt(xyz, src_white='D50', dest_white='D65')
    return xyz

''' white reference 'D65' '''
def read_tiff_16bit_img_into_LAB(tiff_fn, exposure=0, normalize_Lab=False):
    xyz = read_tiff_16bit_img_into_XYZ(tiff_fn, exposure)
    lab = color.xyz2lab(xyz)
    if normalize_Lab:
        normalize_Lab_image(lab)
    return lab

'''
def saveLabIntoTiff16bitImg(tiff_fn,imgLab):
    imgXyz=color.lab2xyz(imgLab)
    imgXyz=XYZ_chromatic_adapt(imgXyz,src_white='D65', dest_white='D50')
    pp_rgb=ProPhotoRGB2XYZ(imgXyz,reverse=True)
    pp_rgb=linearize_ProPhotoRGB(pp_rgb,reverse=True)
    pp_rgb=np.round(pp_rgb*(2**16-1.0))
    idx=np.nonzero(pp_rgb<0)
    pp_rgb[idx[0],idx[1],idx[2]]=0
    idx=np.nonzero(pp_rgb>=(2**16))
    pp_rgb[idx[0],idx[1],idx[2]]=(2**16-1)
    tiff.imsave(tiff_fn,np.uint16(pp_rgb))
'''
    
# white reference 'D65'
def read_tiff_16bit_img_into_sRGB(tiff_fn, exposure=0):
    xyz = read_tiff_16bit_img_into_XYZ(tiff_fn, exposure)
    sRGb = color.xyz2rgb(xyz)
    return sRGb 

'''
def read_sRGB_8bit_img(sRGB_fn):
    # already undo gamma correction in scipy.misc.imread
    sRGB = scipy.misc.imread(sRGB_fn)
    sRGB = np.single(sRGB) / (2 ** 8 - 1.0)
    return sRGB

# white reference 'D65'
def save_image_sRGB_into_sRGB(img_path, srgb):
    srgb = np.float64(srgb)
    idx1, idx2 = np.nonzero(srgb < 0), np.nonzero(srgb > 1)
    srgb[idx1[0], idx1[1], idx1[2]] = 0
    srgb[idx2[0], idx2[1], idx2[2]] = 1    
    scipy.misc.imsave(img_path, srgb)
'''

''' white reference 'D65' '''
def save_image_LAB_into_sRGB(img_path, lab):
    lab = np.float64(lab)
    srgb = color.lab2rgb(lab)
    # clamping to [0,1]
    idx1, idx2 = np.nonzero(srgb < 0), np.nonzero(srgb > 1)
    srgb[idx1[0], idx1[1], idx1[2]] = 0
    srgb[idx2[0], idx2[1], idx2[2]] = 1
    scipy.misc.imsave(img_path, srgb)

def clamp_lab_img(img_lab):
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    l_id_1, l_id_2 = np.nonzero(l < 0), np.nonzero(l > 100)
    a_id_1, a_id_2 = np.nonzero(a < -128), np.nonzero(a > 128)
    b_id_1, b_id_2 = np.nonzero(b < -128), np.nonzero(b > 128)
    img_lab[l_id_1[0], l_id_1[1], 0] = 0
    img_lab[l_id_2[0], l_id_2[1], 0] = 100
    img_lab[a_id_1[0], a_id_1[1], 1] = -128
    img_lab[a_id_2[0], a_id_2[1], 1] = 128
    img_lab[b_id_1[0], b_id_1[1], 0] = -128
    img_lab[b_id_2[0], b_id_2[1], 0] = 128
    return img_lab

def read_img(img_path):
    rgb = scipy.misc.imread(img_path)
    lab = color.rgb2lab(rgb)
    return np.single(lab)
    

def save_img(img_path, lab):   
    lab = clamp_lab_img(lab)    
    rgb = color.lab2rgb(np.float64(lab))
    print 'rgb min max', np.min(rgb[:, :, 0]), np.max(rgb[:, :, 0]), \
    np.min(rgb[:, :, 1]), np.max(rgb[:, :, 1]), np.min(rgb[:, :, 2]), np.max(rgb[:, :, 2])
    
    scipy.misc.imsave(img_path, rgb)

def get_central_pixel_raw_color(batch_img, nb_hs):
    ch, h, w, num_imgs = \
    batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]
    ct_x, ct_y = w / 2, h / 2    
    ct_pix_color = batch_img[:, ct_y, ct_x, :]
    return ct_pix_color;

def get_central_pixel_moments(batch_img, nb_hs):
    nb_sz = 2 * nb_hs + 1
    ch, h, w, num_imgs = \
    batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]
    ct_x, ct_y = w / 2, h / 2

    # compute mean
    mean = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            mean += batch_img[:, ct_y + dy, ct_x + dx, :]
    mean /= nb_sz ** 2

    # standard deviation
    std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            std_dev += np.square(batch_img[:, ct_y + dy, ct_x + dx, :] - mean)
    std_dev = np.sqrt(std_dev / (nb_sz ** 2 - 1))
      
    # skewness
    skew = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            skew += np.power(np.divide(batch_img[:, ct_y + dy, ct_x + dx, :] - mean, std_dev), 3)
    skew = np.nan_to_num(skew)
    skew /= nb_sz ** 2
      
    return np.concatenate((mean, std_dev, skew), axis=0)

def get_central_pixel_grad_vector(batch_img, nb_hs):
    nb_sz = 2 * nb_hs + 1
    ch, h, w, num_imgs = \
    batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]
    ct_x, ct_y = w / 2, h / 2
        
    # 2D gradient vector for each RGB (or LAB channel)
    right, left = batch_img[:, ct_y, ct_x + 1, :], batch_img[:, ct_y, ct_x - 1, :]
    top, bottom = batch_img[:, ct_y + 1, ct_x, :], batch_img[:, ct_y - 1, ct_x, :]
    x_grad, y_grad = (right - left) / np.single(2.0), (top - bottom) / np.single(2.0)
    
    return np.concatenate((x_grad, y_grad), axis=0) 

def get_central_pixel_block_correlation(batch_img, nb_hs):
    nb_sz = 2 * nb_hs + 1
    ch, h, w, num_imgs = \
    batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]
    ct_x, ct_y = w / 2, h / 2
    
    # for central pixel, compute RGB correlation between central patch and left/right/up/down 1-pixel shifted patch
    c_mean = np.zeros((ch, num_imgs), dtype=np.single)
    r_mean = np.zeros((ch, num_imgs), dtype=np.single)
    l_mean = np.zeros((ch, num_imgs), dtype=np.single)
    t_mean = np.zeros((ch, num_imgs), dtype=np.single)
    b_mean = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c_mean += batch_img[:, ct_y + dy, ct_x + dx, :]
            r_mean += batch_img[:, ct_y + dy, ct_x + dx + 1, :]
            l_mean += batch_img[:, ct_y + dy, ct_x + dx - 1, :]
            t_mean += batch_img[:, ct_y + dy + 1, ct_x + dx, :]
            b_mean += batch_img[:, ct_y + dy - 1, ct_x + dx, :]
    c_mean /= nb_sz ** 2
    r_mean /= nb_sz ** 2
    l_mean /= nb_sz ** 2
    t_mean /= nb_sz ** 2
    b_mean /= nb_sz ** 2
      
    c_std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    r_std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    l_std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    t_std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    b_std_dev = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c_std_dev += np.square(batch_img[:, ct_y + dy, ct_x + dx, :] - c_mean)
            r_std_dev += np.square(batch_img[:, ct_y + dy, ct_x + dx + 1, :] - r_mean)
            l_std_dev += np.square(batch_img[:, ct_y + dy, ct_x + dx - 1, :] - l_mean)
            t_std_dev += np.square(batch_img[:, ct_y + dy + 1, ct_x + dx, :] - t_mean)
            b_std_dev += np.square(batch_img[:, ct_y + dy - 1, ct_x + dx, :] - b_mean)    
      
    cr_corre = np.zeros((ch, num_imgs), dtype=np.single)
    cl_corre = np.zeros((ch, num_imgs), dtype=np.single)
    ct_corre = np.zeros((ch, num_imgs), dtype=np.single)
    cb_corre = np.zeros((ch, num_imgs), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c = np.divide(batch_img[:, ct_y + dy, ct_x + dx, :] - c_mean, c_std_dev)
            r = np.divide(batch_img[:, ct_y + dy, ct_x + dx + 1, :] - r_mean, r_std_dev)
            l = np.divide(batch_img[:, ct_y + dy, ct_x + dx - 1, :] - l_mean, l_std_dev)
            t = np.divide(batch_img[:, ct_y + dy + 1, ct_x + dx, :] - t_mean, t_std_dev)
            b = np.divide(batch_img[:, ct_y + dy - 1, ct_x + dx, :] - b_mean, b_std_dev)
            cr_corre += c * r
            cl_corre += c * l
            ct_corre += c * t
            cb_corre += c * b
    cr_corre = np.nan_to_num(cr_corre / (nb_sz ** 2 - 1))
    cl_corre = np.nan_to_num(cl_corre / (nb_sz ** 2 - 1))
    ct_corre = np.nan_to_num(ct_corre / (nb_sz ** 2 - 1))
    cb_corre = np.nan_to_num(cb_corre / (nb_sz ** 2 - 1))

    return np.concatenate((cr_corre, cl_corre, ct_corre, cb_corre), axis=0) 

# for each image, compute 
# 1) 3rd order color moments of the central pixel
# 2) 2d gradient vector
# 3) center to left/right/top/bottom color correlation
# batch_img: shape (ch,h,w,num_imgs)
def get_central_pixel_feature(batch_img, nb_hs):
    nb_sz = 2 * nb_hs + 1
    ch, h, w, num_imgs = \
    batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]
    ct_x, ct_y = w / 2, h / 2
    
    # extrac central pixel color
    ct_pix_color = batch_img[:, ct_y, ct_x, :]
    return ct_pix_color;


# append 'reflected' pixels on the image boundary to pad input images
def get_expanded_img(img, ap_width):
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    expanded_img = np.zeros((h + 2 * ap_width, w + 2 * ap_width, ch), dtype=np.single)

    expanded_img[ap_width:ap_width + h, ap_width:ap_width + w, :] = img
    # four expanded rows/columns. take reflection
    rows_top = img[:ap_width, :, :];
    expanded_img[:ap_width, ap_width:ap_width + w, :] = rows_top[::-1, :, :]
    row_bottom = img[h - ap_width:h, :, :]
    expanded_img[ap_width + h:ap_width * 2 + h, ap_width:ap_width + w, :] = row_bottom[::-1, :, :]
    col_left = img[:, :ap_width, :]
    expanded_img[ap_width:ap_width + h, 0:ap_width, :] = col_left[:, ::-1, :]
    col_right = img[:, w - ap_width:w, :]
    expanded_img[ap_width:ap_width + h, ap_width + w:ap_width * 2 + w, :] = col_right[:, ::-1, :]
    # four corners. Take reflection
    left_top_corn = img[:ap_width, :ap_width, :]
    expanded_img[:ap_width, :ap_width, :] = left_top_corn[::-1, ::-1, :]
    right_top_corn = img[:ap_width, w - ap_width:w, :]
    expanded_img[:ap_width, ap_width + w:ap_width * 2 + w, :] = right_top_corn[::-1, ::-1, :]
    left_bottom_corn = img[h - ap_width:h, :ap_width, :]
    expanded_img[ap_width + h:ap_width * 2 + h, :ap_width, :] = left_bottom_corn[::-1, ::-1, :]
    right_bottom_corn = img[h - ap_width:h, w - ap_width:w, :]
    expanded_img[ap_width + h:ap_width * 2 + h, ap_width + w:ap_width * 2 + w, :] = right_bottom_corn[::-1, ::-1, :]
    
    return expanded_img

def get_color_moment(img, nb_hs, ret=None):
    stTime = time.time()
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    nb_sz = 2 * nb_hs + 1
    assert ch == 3
    
    expanded_img = get_expanded_img(img, nb_hs)
    
    # mean
    if ret == None:
        mean = np.zeros((h, w, ch), dtype=np.single)
    else:
        assert ret.shape[0] == h and ret.shape[1] == w and ret.shape[2] == CENTRAL_PX_MOMENTS_DIM
        mean = ret[:, :, :3]
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            mean += expanded_img[nb_hs + dy:nb_hs + dy + h, nb_hs + dx:nb_hs + dx + w, :]
    mean /= nb_sz ** 2
        
    # standard deviation
    accu = np.zeros((h, w, ch), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            accu += \
            (expanded_img[nb_hs + dy:nb_hs + dy + h, nb_hs + dx:nb_hs + dx + w, :] - mean) ** 2
    # corrected sample variance/standard deviation
    if ret == None:
        std_dev = np.sqrt(accu / (nb_sz ** 2 - 1))
    else:
        ret[:, :, 3:6] = np.sqrt(accu / (nb_sz ** 2 - 1))
        
    std_dev_2 = np.sqrt(accu / (nb_sz ** 2))
    idx = np.nonzero(std_dev_2 == 0)
    std_dev_2[idx[0], idx[1], idx[2]] = 1  # avoid zero division
    
    # skewness
    if ret == None:
        skew = np.zeros((h, w, ch), dtype=np.single)
    else:
        skew = ret[:, :, 6:9]
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            skew += \
            np.divide(expanded_img[nb_hs + dy:nb_hs + dy + h, nb_hs + dx:nb_hs + dx + w, :] - mean, \
                      std_dev_2) ** 3
    skew /= nb_sz ** 2
    skew = np.nan_to_num(skew)
    
    elapsed_tm = time.time() - stTime
    
    if ret == None:
        return np.concatenate((mean, std_dev, skew), axis=2)
    else:
        return ret

def get_color_gradient(img, ret=None, central_diff=False):
    stTime = time.time()
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    assert ch == 3
    
    right = np.concatenate((img[:, 1:w, :], img[:, w - 1, :].reshape((h, 1, ch))), axis=1)
    left = np.concatenate((img[:, 0, :].reshape((h, 1, ch)), img[:, 0:(w - 1), :]), axis=1)
    top = np.concatenate((img[1:h, :, :], img[h - 1, :, :].reshape((1, w, ch))), axis=0)
    bottom = np.concatenate((img[0, :, :].reshape((1, w, ch)), img[0:(h - 1), :, :]), axis=0)
    if ret == None:
        if central_diff:
            x_grad = (right - left) * 0.5
            y_grad = (top - bottom) * 0.5
        else:
            x_grad = img - left
            y_grad = img - bottom
    else:
        if central_diff:
            ret[:, :, :3] = (right - left) * 0.5
            ret[:, :, 3:6] = (top - bottom) * 0.5
        else:
            ret[:, :, :3] = (right - img)
            ret[:, :, 3:6] = (top - img)           
    
    
    elapsed_tm = time.time() - stTime
#     print 'get_color_gradient elapsed time:%f' % elapsed_tm
    if ret == None: 
        return np.concatenate((x_grad, y_grad), axis=2)
    else:
        return ret

# for all pixels within image, compute RGB correlation between central patch and left/right/up/down 1-pixel shifted patch
def get_color_correlation(img, nb_hs):
    stTime = time.time()    
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    nb_sz = 2 * nb_hs + 1    
    h2, w2 = h - nb_hs * 2 - 2, w - nb_hs * 2 - 2  # correlation between neighboring pixels (left,right,top,bottom)
    
    # compute mean
    c_mean = np.zeros((h2, w2, ch), dtype=np.single)
    l_mean = np.zeros((h2, w2, ch), dtype=np.single)
    r_mean = np.zeros((h2, w2, ch), dtype=np.single)
    t_mean = np.zeros((h2, w2, ch), dtype=np.single)
    b_mean = np.zeros((h2, w2, ch), dtype=np.single)
    
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c_mean += img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :]
            l_mean += img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx - 1:w - nb_hs - 1 + dx - 1, :]
            r_mean += img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx + 1:w - nb_hs - 1 + dx + 1, :]
            t_mean += img[nb_hs + 1 + dy + 1:h - nb_hs - 1 + dy + 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :]
            b_mean += img[nb_hs + 1 + dy - 1:h - nb_hs - 1 + dy - 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :]
    c_mean /= nb_sz ** 2
    l_mean /= nb_sz ** 2
    r_mean /= nb_sz ** 2
    t_mean /= nb_sz ** 2
    b_mean /= nb_sz ** 2
    

    # compute standard deviation
    c_std_dev = np.zeros((h2, w2, ch), dtype=np.single)
    l_std_dev = np.zeros((h2, w2, ch), dtype=np.single)
    r_std_dev = np.zeros((h2, w2, ch), dtype=np.single)
    t_std_dev = np.zeros((h2, w2, ch), dtype=np.single)
    b_std_dev = np.zeros((h2, w2, ch), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c_std_dev += np.square(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - c_mean)
            l_std_dev += np.square(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx - 1:w - nb_hs - 1 + dx - 1, :] - l_mean)
            r_std_dev += np.square(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx + 1:w - nb_hs - 1 + dx + 1, :] - r_mean)
            t_std_dev += np.square(img[nb_hs + 1 + dy + 1:h - nb_hs - 1 + dy + 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - t_mean)
            b_std_dev += np.square(img[nb_hs + 1 + dy - 1:h - nb_hs - 1 + dy - 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - b_mean)
    c_std_dev = np.sqrt(c_std_dev / (nb_sz ** 2 - 1))
    l_std_dev = np.sqrt(l_std_dev / (nb_sz ** 2 - 1))
    r_std_dev = np.sqrt(r_std_dev / (nb_sz ** 2 - 1))
    t_std_dev = np.sqrt(t_std_dev / (nb_sz ** 2 - 1))
    b_std_dev = np.sqrt(b_std_dev / (nb_sz ** 2 - 1))
    
 
    # compute correlation
    c_l_corre = np.zeros((h2, w2, ch), dtype=np.single)
    c_r_corre = np.zeros((h2, w2, ch), dtype=np.single)
    c_t_corre = np.zeros((h2, w2, ch), dtype=np.single)
    c_b_corre = np.zeros((h2, w2, ch), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            c = np.divide(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - c_mean, c_std_dev)
            l = np.divide(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx - 1:w - nb_hs - 1 + dx - 1, :] - l_mean, l_std_dev)
            r = np.divide(img[nb_hs + 1 + dy:h - nb_hs - 1 + dy, nb_hs + 1 + dx + 1:w - nb_hs - 1 + dx + 1, :] - r_mean, r_std_dev)
            t = np.divide(img[nb_hs + 1 + dy + 1:h - nb_hs - 1 + dy + 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - t_mean, t_std_dev)
            b = np.divide(img[nb_hs + 1 + dy - 1:h - nb_hs - 1 + dy - 1, nb_hs + 1 + dx:w - nb_hs - 1 + dx, :] - b_mean, b_std_dev)
            
            c_l_corre += c * l
            c_r_corre += c * r
            c_t_corre += c * t
            c_b_corre += c * b
    c_l_corre /= nb_sz ** 2 - 1
    c_r_corre /= nb_sz ** 2 - 1
    c_t_corre /= nb_sz ** 2 - 1
    c_b_corre /= nb_sz ** 2 - 1
            
    elapsed_tm = time.time() - stTime
    print 'get_color_correlation_3 elapsed time:%f' % elapsed_tm
    corre = np.zeros((h, w, ch * 4))
    corre[nb_hs + 1:h - nb_hs - 1, nb_hs + 1:w - nb_hs - 1, :] = np.concatenate((c_l_corre, c_r_corre, c_t_corre, c_b_corre), axis=2)
    corre = np.nan_to_num(corre)
    return corre



# get features for all pixels in the image 
def get_pixel_feature(img, nb_hs, ret=None):
    if ret == None:
        color = img
        return color
    else:
        assert ret.shape[2] == CENTRAL_PX_FEATURE_DIM
        ret[:, :, :CENTRAL_PX_COLOR_DIM] = img
        return ret


''' get features for all pixel in the input/output(enhanced) image '''
def get_img_pixels(in_args):
    img_nm, paras = in_args[0], in_args[1] 
    
    in_img_dir, out_img_dir, fredo_image_processing\
    = paras['in_img_dir'], paras['enh_img_dir'], \
    paras['fredo_image_processing']
    in_img_path = os.path.join(in_img_dir, img_nm + '.tif')
    
    if fredo_image_processing == 1:
        in_img_px = read_tiff_16bit_img_into_LAB(in_img_path, 1.5, False)
    else:
        in_img_px = read_tiff_16bit_img_into_LAB(in_img_path)
    return in_img_px
        
def get_color_moments_v2(img, pos_x, pos_y, nb_hs, ret=None):
    assert(pos_x.shape == pos_y.shape)
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]
    nb_sz = 2 * nb_hs + 1
    num = pos_x.shape[0]
    
    expanded_img = get_expanded_img(img, nb_hs)

    if ret == None:
        mean = np.zeros((num, ch), dtype=np.single)
        std_dev = np.zeros((num, ch), dtype=np.single)
        skew = np.zeros((num, ch), dtype=np.single) 
    else:
        assert ret.shape[1] == CENTRAL_PX_MOMENTS_DIM
        ret[:, :] = 0
        mean = ret[:, :3]
        std_dev = ret[:, 3:6]
        skew = ret[:, 6:9]
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            mean += expanded_img[pos_y + nb_hs + dy, pos_x + nb_hs + dx, :]
            
    mean /= nb_sz ** 2
    
    accu = np.zeros((num, ch), dtype=np.single)
    for dy in range(-nb_hs, nb_hs + 1):
        for dx in range(-nb_hs, nb_hs + 1):
            accu += (expanded_img[pos_y + nb_hs + dy, pos_x + nb_hs + dx, :] - mean) ** 2
    std_dev[:, :] = np.sqrt(accu / (nb_sz ** 2 - 1))
    std_dev_2 = np.sqrt(accu / (nb_sz ** 2))
    idx = np.nonzero(std_dev_2)
    std_dev_2[idx[0], idx[1]] = 1
    
    for dy in range(-nb_hs, nb_hs):
        for dx in range(-nb_hs, nb_hs):
            skew += \
            np.divide(expanded_img[pos_y + nb_hs + dy, pos_x + nb_hs + dx, :] - mean, std_dev_2) ** 3
    skew /= nb_sz ** 2
    skew[:, :] = np.nan_to_num(skew)
    
#     print 'mean std_dev skew shape', mean.shape, std_dev.shape, skew.shape
    if ret == None:
        return np.concatenate((mean, std_dev, skew), axis=1)
    else:
        return ret   

def get_color_gradient_v2(img, pos_x, pos_y, ret=None):
    h, w, ch = img.shape[0], img.shape[1], img.shape[2]    
    right = np.concatenate((img[:, 1:w, :], img[:, w - 1, :].reshape((h, 1, ch))), axis=1)
    left = np.concatenate((img[:, 0, :].reshape((h, 1, ch)), img[:, 0:w - 1, :]), axis=1)
    top = np.concatenate((img[1:h, :, :], img[h - 1, :, :].reshape((1, w, ch))), axis=0)
    bottom = np.concatenate((img[0, :, :].reshape((1, w, ch)), img[0:h - 1, :, :]), axis=0)
    
    if ret == None:
        x_grad = (right[pos_y, pos_x, :] - left[pos_y, pos_x, :]) * 0.5
        y_grad = (top[pos_y, pos_x, :] - bottom[pos_y, pos_x, :]) * 0.5
        return np.concatenate((x_grad, y_grad), axis=1)
    else:
        assert ret.shape[1] == CENTRAL_PX_GRAD_VECTOR_DIM
        ret[:, 0:3] = (right[pos_y, pos_x, :] - left[pos_y, pos_x, :]) * 0.5
        ret[:, 3:6] = (top[pos_y, pos_x, :] - bottom[pos_y, pos_x, :]) * 0.5
        return ret
# accept a list pixels position
def get_pixel_feature_v2(img, pos_x, pos_y, ret=None):
    if ret == None:
        color = img[pos_y, pos_x, :]
        return color
    else:
        assert ret.shape[1] == CENTRAL_PX_FEATURE_DIM
        ret[:, :CENTRAL_PX_COLOR_DIM] = img[pos_y, pos_x, :]
        return ret

def getPixLocalContextV2Helper(in_args):
    pix_x, pix_y, integral_map, paras, ftr = \
    in_args[0], in_args[1], in_args[2], in_args[3], in_args[4]
    
    ftr_dim, offsets, hist_bin_num, area_pix_num = \
    paras['ftr_dim'], paras['offsets'], paras['label_num'], paras['area_pix_num']
    num_pixs = pix_x.shape[0]
    num_hist = ftr_dim / hist_bin_num
    
    st_time = time.time()
    for i in range(num_pixs):
        left = offsets[0, :] + pix_x[i] - 1
        top = offsets[1, :] + pix_y[i] - 1
        right = offsets[2, :] + pix_x[i]
        bottom = offsets[3, :] + pix_y[i]
        ftr[i, :] = integral_map[bottom, right] + integral_map[top, left]\
        - integral_map[bottom, left] - integral_map[top, right]
    ftr /= area_pix_num[np.newaxis, :]
    return ftr

''' pixX,pixX should be pre-increased by 100 (appending width) '''
def getPixContextSem(in_args):
    pix_x, pix_y, integral_map, paras, pool, ftr = \
    in_args[0], in_args[1], in_args[2], in_args[3], in_args[4], in_args[5]
    
    ftr_dim, offsets, hist_bin_num, area_pix_num = \
    paras['ftr_dim'], paras['offsets'], paras['label_num'], paras['area_pix_num']
    num_pixs = pix_x.shape[0]
    
    if not pool == None:
        num_parts = 16
        part_size = num_pixs / num_parts
        start = range(0, part_size * num_parts, part_size)
        end = [s + part_size for s in start]
        end[num_parts - 1] = num_pixs
        assert len(start) == num_parts
        pix_x2, pix_y2 = [None] * num_parts, [None] * num_parts
        for i in range(num_parts):
            pix_x2[i] = pix_x[start[i]:end[i]]
            pix_y2[i] = pix_y[start[i]:end[i]]
        
        func_args = zip(pix_x2, pix_y2, [integral_map] * num_parts, [paras] * num_parts, \
                        [None] * num_parts, [None] * num_parts)
        try:
            results = pool.map(getPixContextSem, func_args, num_parts / 4)
        except UtilImageError, e:
            print e        
        for i in range(num_parts):
            ftr[start[i]:end[i], :] = results[i]
    else:
        if ftr == None:
            ftr = np.zeros((num_pixs, ftr_dim), dtype=np.single)
        for i in range(num_pixs):
            left = offsets[0, :] + pix_x[i] - 1
            top = offsets[1, :] + pix_y[i] - 1
            right = offsets[2, :] + pix_x[i]
            bottom = offsets[3, :] + pix_y[i]
            for j in range(hist_bin_num):
                i_map = integral_map[j]
                ftr[i, j::hist_bin_num] = i_map[bottom, right] + i_map[top, left]\
                - i_map[bottom, left] - i_map[top, right]   
                ftr[i, j::hist_bin_num] = ftr[i, j::hist_bin_num] / area_pix_num
    return ftr



''' use cpu parallel computing to get pixel local context
    one histogram bin -> one thread '''
def getPixLocContextV2(in_args):
    pix_x, pix_y, integral_map, paras, pool, ftr, num_proc = \
    in_args[0], in_args[1], in_args[2], in_args[3], in_args[4], in_args[5], in_args[6]

    ftr_dim, offsets, hist_bin_num, area_pix_num = \
    paras['ftr_dim'], paras['offsets'], paras['label_num'], paras['area_pix_num']
    num_pixs = pix_x.shape[0]
    num_hist = ftr_dim / hist_bin_num
    assert num_hist * hist_bin_num == ftr_dim
    
    ftr = ftr.reshape((num_pixs, num_hist, hist_bin_num))
    l_ftr = [None] * hist_bin_num
    for i in range(hist_bin_num):
        l_ftr[i] = ftr[:, :, i]
    
    func_args = zip([pix_x] * hist_bin_num, [pix_y] * hist_bin_num, integral_map, [paras] * hist_bin_num, l_ftr)
    num_proc = hist_bin_num if num_proc > hist_bin_num else num_proc
    try:
        results = pool.map(getPixLocalContextV2Helper, func_args, hist_bin_num / num_proc)
    except UtilImageError, e:
        print 'exception is found'
        print e
    for i in range(hist_bin_num):
        ftr[:, :, i] = results[i]          
    ftr = ftr.reshape((num_pixs, num_hist * hist_bin_num))
  

def get_pixel_local_context_mean_color_helper(in_args):
    '''ftr shape: (num_pixs,pool_region_num,3)'''
    pix_x, pix_y, color_itg_map, paras, ftr = \
    in_args[0], in_args[1], in_args[2], in_args[3], in_args[4]
    
    num_pixs = pix_x.shape[0]
    
    offsets, area_pix_num = paras['offsets'], paras['area_pix_num']
    for i in range(num_pixs):
        left = offsets[0, :] + pix_x[i] - 1
        top = offsets[1, :] + pix_y[i] - 1
        right = offsets[2, :] + pix_x[i]
        bottom = offsets[3, :] + pix_y[i]
        ftr[i, :, :] = color_itg_map[bottom, right,:] + color_itg_map[top, left,:]\
        - color_itg_map[bottom, left,:] - color_itg_map[top, right,:]
    ftr /= area_pix_num[np.newaxis, :,np.newaxis]
    return ftrgetPixContextColorFtr
    
def get_pixel_local_context_mean_color(in_args):
    pix_x, pix_y, color_itg_map, paras, pool, ftr = \
    in_args[0], in_args[1], in_args[2], in_args[3], in_args[4], in_args[5]
    
    offsets, pool_region_num, area_pix_num = \
    paras['offsets'], paras['pool_region_num'], paras['area_pix_num']
    num_pixs = pix_x.shape[0]
    
    
    ftr = ftr.reshape((num_pixs, pool_region_num, 3))
    l_ftr = [None] * 3
    for i in range(3):
        l_ftr[i] = ftr[:, :, i]
    
    func_args = zip([pix_x] * 3, [pix_y] * 3, color_itg_map, [paras] * 3, \
                    l_ftr)
    try:
        results = pool.map(get_pixel_local_context_mean_color_helper, func_args)
    except UtilImageError, e:
        print e
    
    for i in range(3):
        ftr[:, :, i] = results[i]
    ftr = ftr.reshape((num_pixs, pool_region_num * 3))

def getPixContextColorFtr(pixX, pixY, colorIntegralMap, paras):  
    offsets, area_pix_num = paras['offsets'], paras['area_pix_num']     
    n_pix, n_region = pixX.shape[0], offsets.shape[1]
    
    colorFtr = np.zeros((n_pix, n_region, 3))
    print 'colorIntegralMap shape',colorIntegralMap.shape
    for i in range(n_pix):
        left = offsets[0, :] + pixX[i] - 1
        top = offsets[1, :] + pixY[i] - 1
        right = offsets[2, :] + pixX[i]
        bottom = offsets[3, :] + pixY[i]
        colorFtr[i, :, :] = colorIntegralMap[bottom, right, :] + colorIntegralMap[top, left, :]\
        - colorIntegralMap[bottom, left, :] - colorIntegralMap[top, right, :]
    colorFtr /= area_pix_num[np.newaxis, :, np.newaxis]
    colorFtr = colorFtr.reshape((n_pix, n_region * 3))
    return colorFtr 

def next_4_multiple(val):
    return ((np.int64(val) + 3) / 4) * 4


QUAD_POLY_COLOR_BASIS_DIM = 10
#     colors shape: num_colors * 3
def quad_poly_color_basis(colors):
    assert colors.shape[1] == 3
    num_color = colors.shape[0]
    basis = np.zeros((num_color, QUAD_POLY_COLOR_BASIS_DIM), dtype=np.single)
    basis[:, 0] = colors[:, 0] * colors[:, 0]
    basis[:, 1] = colors[:, 1] * colors[:, 1]
    basis[:, 2] = colors[:, 2] * colors[:, 2]
    basis[:, 3] = colors[:, 0] * colors[:, 1]
    basis[:, 4] = colors[:, 0] * colors[:, 2]
    basis[:, 5] = colors[:, 1] * colors[:, 2]
    basis[:, 6] = colors[:, 0]
    basis[:, 7] = colors[:, 1]
    basis[:, 8] = colors[:, 2]
    basis[:, 9] = 1
    return basis

def quad_poly_patch_color_basis(patch):
    h, w, c = patch.shape[0], patch.shape[1], patch.shape[2]
    assert c == 3
    basis = np.zeros((h, w, 10), dtype=np.single)
    for i in range(h):
        for j in range(w):
            l, a, b = np.single(patch[i, j, 0]), np.single(patch[i, j, 1]), np.single(patch[i, j, 2])
            basis[i, j, :] = [l * l, a * a, b * b, l * a, l * b, a * b, l, a, b, 1]
    return basis    

# return a histogram of lightness in the range (0,100)
def get_img_lightness_hist(L, range_min=0, range_max=100, bins=50):
    h, w = L.shape[0], L.shape[1]
    L1 = scipy.ndimage.filters.gaussian_filter(L, sigma=10, order=0)
    L2 = scipy.ndimage.filters.gaussian_filter(L, sigma=20, order=0)
    
    hist, bin_edges = np.histogram(L.flatten(), bins, range=(range_min, range_max), normed=False)
    hist = np.single(hist) / np.single(h * w)
    hist1, bin_edges1 = np.histogram(L1.flatten(), bins, range=(range_min, range_max), normed=False)
    hist1 = np.single(hist1) / np.single(h * w)
    hist2, bin_edges2 = np.histogram(L2.flatten(), bins, range=(range_min, range_max), normed=False)
    hist2 = np.single(hist2) / np.single(h * w)    
    return np.concatenate((hist, hist1, hist2))

def get_img_scene_brightness(L, img_path):
    img_f = open(img_path, 'rb')
    tags = exifread.process_file(img_f, details=False)
    print 'EXIF tags keys', tags.keys()
    
    if 'ExposureTime' in tags.keys():
        exposure_time = tags['EXIF ExposureTime']
        exposure_time_str = "%s" % exposure_time
        print 'exposure_time_str', exposure_time_str
        idx = exposure_time_str.find(r'/')
        if idx != -1:
            exposure_time = float(exposure_time_str[:idx]) / float(exposure_time_str[idx + 1:])    
        else:
            exposure_time = float(exposure_time_str)
    else:
        print 'set exposure_time'
        exposure_time = 1.0 / 60.0
        
        
    if 'EXIF FNumber' in tags.keys():
        f_stop = tags['EXIF FNumber']
        f_stop_str = "%s" % f_stop
        idx = f_stop_str.find(r'/')
        if  idx != -1:
    #         print f_stop_str[:idx],f_stop_str[idx+1:]
            val1 = float(f_stop_str[:idx])
            val2 = float(f_stop_str[idx + 1:])
            print 'f_stop_str val1 val2', f_stop_str, val1, val2
            f_stop = val1 / val2
        else:
            f_stop = float(f_stop_str)
    else:
        print 'set f stop to default value'        
        f_stop = 10.0
    
    if 'EXIF ISOSpeedRatings' in tags.keys():
        iso = tags['EXIF ISOSpeedRatings']
        iso_str = "%s" % iso
        iso = float(iso_str)
    else:
        print 'set iso to default value'
        iso = 200
        
    med_L = np.median(L.flatten())
    
    print 'exposure time:%f f_stop:%f iso:%f median L:%f' % \
    (exposure_time, f_stop, iso, med_L)    
    
    return med_L / (f_stop * f_stop * exposure_time * iso)

def get_BSpline_curve(x, y, num_control_points, t_min, t_max):
    p = 3  # degree is 3
    n = num_control_points - 1
    m = p + n + 1  # (m+1) knots, irst p (last p) knots are the same  
    t = np.linspace(t_min, t_max, m - p - p + 1)
    t = t[1:-1]
    spline = scipy.interpolate.LSQUnivariateSpline(x, y, t)
    return spline

# get a B-spline for cumulative probability functioin of histogram
def get_cum_hist_BSpline_curve(cumsum_hist, t_min, t_max, bins):
    p = 3  # degree is 3
    n = 50  # 51 control points
    m = p + n + 1  # (m+1) knots, first p (last p) knots are the same    
    step = (t_max - t_min) / bins
    t = np.linspace(t_min, t_max, m - p - p + 1)
    assert len(t) == (m - p - p + 1)
    t = t[1:-1]
    x = np.linspace(t_min + step * 0.5, t_max - step * 0.5, bins)
    spline = scipy.interpolate.LSQUnivariateSpline(x, cumsum_hist, t)
    return spline, x
    
# return 51 control points with even spaced know vector. Degree of b spline is 3
def get_lightness_equalization_curve_control_points(L):
    h, w = L.shape[0], L.shape[1]
    bins = 100
    hist, bin_edges = np.histogram(L, bins, range=(0, 100), density=False)
    hist = np.single(hist) / np.single(h * w)
    cumsum_hist = np.cumsum(hist)
    spline, x = get_cum_hist_BSpline_curve(cumsum_hist, 0.0, 100.0, bins)
#     mpplot.plot(x,cumsum_hist)
#     mpplot.plot(x, spline(x),'.-')
#     mpplot.show()
    return spline.get_coeffs()
#     print eq_curve

 
# sigma: std_dev of gaussian derivative filter
def get_lightness_detail_weighted_equalization_curve_control_points(L, sigma):
    bins = 100
    grad_mag = scipy.ndimage.filters.gaussian_gradient_magnitude(L, sigma)
    hist, bin_edges = np.histogram(L, bins, range=(0, 100), weights=grad_mag)
    hist = np.single(hist) / np.sum(grad_mag)
    cumsum_hist = np.cumsum(hist)
    spline, x = get_cum_hist_BSpline_curve(cumsum_hist, 0.0, 100.0, bins)
    return spline.get_coeffs()        

def get_highlight_clipping_value(L, percentage):
    h, w = L.shape[0], L.shape[1]
    rank = np.round(h * w * percentage)
    sorted_L = np.sort(L.flatten())
#     print 'sorted_L',sorted_L[:10],sorted_L[-10:]
    return sorted_L[-rank]

def get_tone_spatial_distribution(L, num_interval):
    edges = np.linspace(0.0, 100.0, num_interval + 1)
    ftr = np.zeros((num_interval, 3), dtype=np.single)
    for i in range(num_interval):
        idx = np.nonzero((L >= edges[i]) & (L < edges[i + 1]))
        num_pix = len(idx[0])
        if num_pix > 0:
            cy = np.mean(idx[0])
            cx = np.mean(idx[1])
            std_dev_y = np.sqrt(np.sum((idx[0] - cy) ** 2) / num_pix)
            std_dev_x = np.sqrt(np.sum((idx[1] - cx) ** 2) / num_pix)
            ftr[i, 0] = std_dev_x * std_dev_y / num_pix
            ftr[i, 1:3] = [cx, cy]            
        else:
            ftr[i, :] = 0

    return ftr.reshape((num_interval * 3))

'''compute histogram of background pixels hue
semMap: (h,w) semantic map
semFgBg: (n) semantic label => {0,1} indicate foreground/background
'''
def getBackgroundHueHist(imgLab,semMap,semFgBg):
    imgHsv=color.rgb2hsv(color.lab2rgb(imgLab))
    assert imgLab.shape[:2] == semMap.shape
    fgBgMap=semFgBg[semMap]
    bgIdx=np.nonzero(fgBgMap==0)
    bgHue=imgHsv[bgIdx[0],bgIdx[1],0]
    hist,binEdges=np.histogram(bgHue, bins=30, range=(0,1))
    hist=np.float32(hist)
    hist/=np.sum(hist)
    return hist

def get_image_global_ftr(in_args):
    img_nm, paras = in_args[0], in_args[1]
    if len(in_args)>2:
        bgHueHist = 1
        semMapFile,semFgBg = in_args[2],in_args[3]
    else:
        bgHueHist = 0
    in_img_dir = paras['in_img_dir']
    fredo_image_processing = paras['fredo_image_processing']
    log_luminance = paras['log_luminance']
    
    if bgHueHist:
        semMap=os.path.join(paras['semantic_map_dir'],semMapFile)
        semMap = scipy.io.loadmat(semMap)
        semMap = semMap['responseMap']

    tif_img_path = os.path.join(in_img_dir, img_nm + '.tif')
    if fredo_image_processing == 1:
        Lab_img = read_tiff_16bit_img_into_LAB(tif_img_path, 1.5)
    else:
        Lab_img = read_tiff_16bit_img_into_LAB(tif_img_path)
    L = Lab_img[:, :, 0]
    h, w = L.shape[0], L.shape[1]
    if log_luminance == 1:
        idx = np.nonzero(L == 0)
        L_copy = np.zeros((h, w), dtype=np.single)
        L_copy[:, :] = L[:, :]
        L_copy[idx[0], idx[1]] = 1
        log_L = np.log(L_copy)
        
    if log_luminance == 1:
        lightness_hist = get_img_lightness_hist(log_L, 0, np.log(100))
    else:
        lightness_hist = get_img_lightness_hist(L, 0, 100)
        
    if 0:
        img_a = Lab_img[:, :, 1]
        img_b = Lab_img[:, :, 2]
        a_hist = get_img_lightness_hist(img_a, -128.0, 128.0)
        b_hist = get_img_lightness_hist(img_b, -128.0, 128.0)
        lightness_hist = np.concatenate((lightness_hist, a_hist, b_hist))
    
    scene_brightness = get_img_scene_brightness(L, tif_img_path)
    cp1 = get_lightness_equalization_curve_control_points(L)
    cp2 = get_lightness_detail_weighted_equalization_curve_control_points(L, 1)
    cp3 = get_lightness_detail_weighted_equalization_curve_control_points(L, 10)
    cp4 = get_lightness_detail_weighted_equalization_curve_control_points(L, 20)
    hl_clipping = np.zeros((6), dtype=np.single)
    hl_clipping[0] = get_highlight_clipping_value(L, 0.01)
    hl_clipping[1] = get_highlight_clipping_value(L, 0.02)
    hl_clipping[2] = get_highlight_clipping_value(L, 0.03)
    hl_clipping[3] = get_highlight_clipping_value(L, 0.05)
    hl_clipping[4] = get_highlight_clipping_value(L, 0.1)
    hl_clipping[5] = get_highlight_clipping_value(L, 0.15)
    L_spatial_distr = get_tone_spatial_distribution(L, 10)
    if 0:
        a_spatial_distr = get_tone_spatial_distribution(img_a, 10)
        b_spatial_distr = get_tone_spatial_distribution(img_b, 10)
        L_spatial_distr = np.concatenate((L_spatial_distr, a_spatial_distr, b_spatial_distr))
    if bgHueHist:
        bgHueHist=getBackgroundHueHist(Lab_img,semMap,semFgBg)
    else:
        bgHueHist=np.zeros((30))

    return lightness_hist, scene_brightness, cp1, cp2, cp3, cp4, \
        hl_clipping, L_spatial_distr, bgHueHist


# dilate outwards by 'extend_width' pixels
def get_extended_edge_pixel(edge_pix_x, edge_pix_y, h, w, extend_width):
    edge_mask = np.zeros((h + 2 * extend_width, w + 2 * extend_width), dtype=np.bool)
    for dy in range(-extend_width, extend_width + 1):
        for dx in range(-extend_width, extend_width + 1):
            l_edge_pix_y, l_edge_pix_x = edge_pix_y + extend_width + dy, edge_pix_x + extend_width + dx
            edge_mask[l_edge_pix_y, l_edge_pix_x] = 1
    edge_mask = edge_mask[extend_width:h + extend_width, extend_width:w + extend_width]
    idx = np.nonzero(edge_mask)
    return np.array(idx[1]), np.array(idx[0])
    
def get_affected_pixels_helper(in_args):
    h, w, seg, dilate_width = \
    in_args[0], in_args[1], in_args[2], in_args[3]
    seg = np.array(seg)
    seg_x, seg_y = seg % w, seg / w
    affected_x, affected_y = get_extended_edge_pixel(seg_x, seg_y, h, w, dilate_width)
    affected_id = affected_y * w + affected_x
    return affected_id
    
def get_affected_pixels(pool, h, w, img_seg, dilate_width):
    num_seg = len(img_seg)
    if pool == None:
        affected_ids = [None] * num_seg
        for j in range(num_seg):
            l_seg = np.array(img_seg[j])
            l_seg_x, l_seg_y = l_seg % w, l_seg / w
            l_included_x, l_included_y = get_extended_edge_pixel(l_seg_x, l_seg_y, h, w, dilate_width)
            affected_ids[j] = l_included_y * w + l_included_x        
    else:
        func_args = zip([h] * num_seg, [w] * num_seg, img_seg, [dilate_width] * num_seg)
        try:
            results = pool.map(get_affected_pixels_helper, func_args)
        except UtilImageError, e:
            print e
        affected_ids = results
    return affected_ids
