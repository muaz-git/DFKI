import json
import numpy as np
import cv2
import scipy
from sklearn.svm import SVC
IMAGE_HEIGHT = 998.0

def preprocessImg(img):
    height, width = img.shape

    img = cv2.GaussianBlur(img, (3, 3), 0, 0.0, 4)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0.0, 4)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 7)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=6)
    img = cv2.GaussianBlur(img, (9, 9), 0, 0.0, 4)
    scaleFactor = IMAGE_HEIGHT / (height)
    n_width = scaleFactor * width
    # img = cv2.resize(img, (int(n_width), int(IMAGE_HEIGHT)))
    return img

def extractFeatures(img):
    img = cv2.resize(img, (28,28),0,0, cv2.INTER_AREA)
    src_f = np.float32(img)

    kernel_size, sig, th, lm, gm, ps = 13, 1, 0, 1.0, 0.02, 0

    kernel0 = cv2.getGaborKernel((kernel_size, kernel_size), sig, th, lm, gm, ps)
    kernel45 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 45, 1.), (kernel_size, kernel_size))
    kernel90 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 90, 1.), (kernel_size, kernel_size))
    kernel135 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 135, 1.), (kernel_size, kernel_size))

    dest0 = cv2.filter2D(src_f, cv2.CV_32F, kernel0)
    dest45 = cv2.filter2D(src_f, cv2.CV_32F, kernel45)
    dest90 = cv2.filter2D(src_f, cv2.CV_32F, kernel90)
    dest135 = cv2.filter2D(src_f, cv2.CV_32F, kernel135)

    # print np.amin(dest0)
    # print np.amax(dest0)
    # scale_factor = 1.0 / 255.0
    scale_factor = 1
    viz0 = np.uint8(dest0)*scale_factor
    viz45 = np.uint8(dest45)*scale_factor
    viz90 = np.uint8(dest90)*scale_factor
    viz135 = np.uint8(dest135)*scale_factor

    print np.amin(viz0)
    print np.amax(viz0)

    viz0 = np.reshape(viz0, (1, np.product(viz0.shape)))
    viz45 = np.reshape(viz45, (1, np.product(viz45.shape)))
    viz90 = np.reshape(viz90, (1, np.product(viz90.shape)))
    viz135 = np.reshape(viz135, (1, np.product(viz135.shape)))

    rowVector = np.append(np.append(np.append(viz0, viz45), viz90), viz135)
    print rowVector.shape
    print rowVector.dtype
    # print kernel0
    # print kernel45
    return img

def formFeatures(img):
    src_f = np.float32(img)
    # print src_f.shape
    kernel_size, sig, th, lm, gm, ps = 13, 1, 0, 1.0, 0.02, 0

    kernel0 = cv2.getGaborKernel((kernel_size, kernel_size), sig, th, lm, gm, ps)
    kernel45 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 45, 1.),
                              (kernel_size, kernel_size))
    kernel90 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 90, 1.),
                              (kernel_size, kernel_size))
    kernel135 = cv2.warpAffine(kernel0, cv2.getRotationMatrix2D((kernel_size / 2., kernel_size / 2.), 135, 1.),
                               (kernel_size, kernel_size))

    dest0 = cv2.filter2D(src_f, cv2.CV_32F, kernel0)
    dest45 = cv2.filter2D(src_f, cv2.CV_32F, kernel45)
    dest90 = cv2.filter2D(src_f, cv2.CV_32F, kernel90)
    dest135 = cv2.filter2D(src_f, cv2.CV_32F, kernel135)

    scaling_factor = 1./255
    dest0 = np.uint8(dest0) * scaling_factor
    dest45 = np.uint8(dest45) * scaling_factor
    dest90 = np.uint8(dest90) * scaling_factor
    dest135 = np.uint8(dest135) * scaling_factor

    feature_set = np.dstack((dest0, dest45, dest90, dest135))
    #
    # cv2.imshow('image', feature_set[:, :, 3])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[5, 6], [7, 8]])
    # c = np.array([[9, 10], [11, 12]])
    # x = np.dstack((a, b, c))
    # np.reshape(x, 12, order='F')
    # array([1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12])

    # print (np.concatenate((dest0, dest45), axis=2)).shape
    # print dest0.shape
    # print dest45.shape
    # print dest90.shape
    # print dest135.shape
    return feature_set

def fetch_samples(feature_set, samples):
    class_1 = 'EPC_EVENT'
    # class_1 = 'EPC_EVENT'

    X = np.array([])
    scale_size = 28
    tot_layers = np.size(feature_set,axis=2)
    tot_dimensions = scale_size * scale_size * tot_layers

    # print 'Total dimensions = '+str(tot_dimensions)
    X = np.empty((0,tot_dimensions), feature_set.dtype)
    Y = []



    for s in samples:
        label = 0
        cropped = feature_set[s['y']:s['y']+s['height'], s['x']:s['x']+s['width'],:]

        img = cv2.resize(cropped, (28, 28), 0, 0, cv2.INTER_AREA)
        X = np.append(X, np.array([np.reshape(img, tot_dimensions, order='F')]), axis=0)

        if s['objectType'] == class_1:
            label = 1

        Y.append(label)

    # print len(samples)
    print X.shape
    print len(Y)
    # cv2.imshow('image', img[:,:,3])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print X.shape
    # print np.array(Y).shape
    return X, np.array(Y)
    # exit()

def applyML(Xtrain, Ytrain, Xtest, Ytest):
    clf = SVC()
    clf.fit(Xtrain, Ytrain)

def processing(fL, data):
    i = 1
    X_complete = None
    Y_complete = None
    for d in data:
        path = d["filePath"]
        samples = d["samples"]
        print path

        img = cv2.imread(fL + path, cv2.CV_8UC1)

        img = preprocessImg(img)

        feature_set = formFeatures(img)
        X, Y = fetch_samples(feature_set, samples)

        if X_complete is None:
            X_complete = X
            Y_complete = Y
        else:
            X_complete = np.concatenate((X_complete, X), axis=0)
            Y_complete = np.concatenate((Y_complete, Y), axis=0)




    print '*****************'
    print X_complete.shape
    print Y_complete.shape

    myDict = {}
    myDict['Xtrain'] = X_complete
    myDict['Ytrain'] = Y_complete
    np.save('/trainingData.npy', myDict)
    dd = np.load('/trainingData.npy', myDict)
    print dd['Xtrain'].shape
    print dd['Ytrain'].shape
    exit()
    # exit()
    # cv2.imshow('image', feature_set[:, :, 0:3])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = extractFeatures(img)
    #
    # # original = img.copy()
    # # original = cv2.bitwise_not(original)
    #
    # # img = cv2.resize(img, None, fx=0.75, fy=0.75)
    #
    # # print img.dtype
    # return
    # exit()
    # # cv2.namedWindow('image', cv2.WINDOW_NORMAL)



folderLocation = "./data/"
with open(folderLocation+'positives.json') as data_file:
    data = json.load(data_file)


processing(folderLocation, data)



# fP = data[0]["filePath"]
# samples = data[0]["samples"]
# processing(folderLocation, fP, samples)


