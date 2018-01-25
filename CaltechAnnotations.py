import os
import glob
from multiprocessing import Pool
import cv2

def getimagefilelist(caltechpath, subset):
    if subset == "train":
        sets = ["set00","set01", "set02", "set03", "set04", "set05"]
    else:
        sets = ["set06", "set07", "set08", "set09", "set10"]

    imagelist = []

    for imgset in sets:
        searchpath = os.path.join(caltechpath, subset, '**', '*.jpg')
        print(searchpath)
        for filename in glob.iglob(searchpath, recursive=True):
            imagelist.append(filename)

    return imagelist


def getannofile(annopath, imagefile):
    dirname = os.path.dirname(imagefile)
    subset = os.path.basename(dirname)
    setname = os.path.basename(os.path.dirname(dirname))
    filename = os.path.basename(imagefile)
    annofilename = '{}_{}_{}.txt'.format(setname, subset, filename)
    annofilename = os.path.join(annopath, annofilename)
    return annofilename


def readannotations(annopath, imagefile):

    img = cv2.imread(imagefile)
    height = img.shape[0]
    width = img.shape[1]
    annofile = getannofile(annopath, imagefile)
    output = {'filepath': imagefile, 'width':width, 'height':height}
    GT = []
    numbb = 0
    for line in open(annofile, 'r'):
        words = line.split(' ')
        if words[0] == 'person':
            bboxcomplete = words[1:5]
            bboxcomplete = list(map(float, bboxcomplete))
            bboxcomplete = list(map(int, bboxcomplete))
            pedheight = bboxcomplete[3]
            if words[5] == '1':
                bboxpartial = words[6:10]
                bboxpartial = list(map(float, bboxpartial))
                bboxpartial = list(map(int, bboxpartial))
                occlusion = float(bboxpartial[3])/float(bboxcomplete[3])
            else:
                bboxpartial = []
                occlusion = 0.

            if (pedheight >=50) and (occlusion <=0.35):
                if bboxpartial == []:
                    bbox  = bboxcomplete
                else:
                    bbox = bboxpartial
                #print(bbox)
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0 :
                    bbox[1] = 0

                x2 = bbox[0] + bbox[2]
                if x2 > width:
                    x2 = width - 1
                y2 = bbox[1] + bbox[3]
                if y2 > height:
                    y2 = height - 1
                res = {'class':'person', 'x1': bbox[0], 'x2' : x2 ,
                       'y1' : bbox[1], 'y2' : y2, 'difficult':
                           False}
                GT.append(res)
                numbb+=1

    if GT != []:
        output['bboxes'] =  GT
        # print(output)
        return output, numbb
    else:
        return None, 0

def getall(caltechpath, subset):
    print('parsing annotations start')
    imagelist = getimagefilelist(caltechpath, subset)
    baseannopath = os.path.join(os.path.dirname(caltechpath), 'annotations')
    if subset == 'train':
        annopath = os.path.join(baseannopath, 'anno_train_1xnew')
    else:
        annopath = os.path.join(baseannopath, 'anno_test_1xnew')

    args = [(annopath, i) for i in imagelist]

    p = Pool(64)

    results = p.starmap(readannotations, args)

    annotations = [x[0] for x in results]
    numbb = [x[1] for x in results]
    results = [x for x in annotations if x is not None]
    print('parsing annotations end')
    return results, {'person':sum(numbb)}, {'person' : 0}


