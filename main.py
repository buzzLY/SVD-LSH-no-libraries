########################################


from pyspark import SparkContext, SparkConf
import sys
import io
import zipfile
import numpy as np
import hashlib
from tifffile import TiffFile
from numpy.linalg import svd

# Convert file content in array
def getOrthoTif(fileName, zfBytes):
    fileName = fileName[fileName.rfind('/') + 1:]
#    Given a zipfile as bytes (i.e. from reading from a binary file),
#    Return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
#    find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':
#            found it, turn into array:
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
            arr = tif.asarray();
#            Divide image in 500 * 500 size
            res = []
            factor = int(arr.shape[0]/500)
            hArr = np.split(arr, factor, axis=0)
            count = 0
            for h in hArr:
                vArr = np.split(h, factor, axis = 1)
                for v in vArr:
                    name = fileName + "-" + str(count)
                    res.append((name, v))
                    count += 1
            return res

# Calculate feature vector in range of -1 and 1
def calFeatureVector(arr):
    outputArray = np.zeros(shape=(500,500), dtype=np.float64)
    for i, a in enumerate(arr):
        for j, val in enumerate(a):
            red, green, blue, infrared = val
            rgb_mean = ((float(red)*0.33+float(green)*0.33+float(blue)*0.33))
            outputArray[i][j] = rgb_mean*(float(infrared)/100.0)
    arr = dimensionalityReduction(outputArray)
    row_diff = np.diff(arr, axis=1).flatten()
    col_diff = np.diff(arr, axis=0).flatten()
    row_diff = np.where(row_diff < -1, -1, (np.where(row_diff > 1, 1, 0)))
    col_diff = np.where(col_diff < -1, -1, (np.where(col_diff > 1, 1, 0)))
    res = np.hstack((row_diff, col_diff))
    return res


# Reduce dimensionality
def dimensionalityReduction(arr):
    outputArray = np.zeros(shape=(50,50), dtype=np.float64)
    arr1 = np.split(arr, 50, axis=0)
    i = j = 0
    for a in arr1:
        x = np.split(a, 50, axis=1)
        j = 0
        for val in x:
            outputArray[i][j] = np.mean(val)
            j += 1
        i += 1
    return outputArray

# Convert feature to 128 bit signature and take 10th bit
def getSignature(featureVector):
    signature = []
    count = 0
    endIndex = 0
    for i in range(0,128):
        startingIndex = endIndex
        if count < 36:
            endIndex = startingIndex + 39
        else:
            endIndex = startingIndex + 38
        count += 1
        val_hex = hashlib.md5(featureVector[startingIndex:endIndex]).hexdigest()
        signature.append(int(bin(int(val_hex, 16))[2:][10]))
    return signature

# Generate hash value for current band
def generateHash(bandVector):
    out = 0
    for bit in bandVector:
        out = (out << 1) | bit
    smoothing = 100
    return int(out % (len(bandVector) * smoothing))

# Apply LSH
def applyLSH(pair):
    key, featureVector = pair
    signature = getSignature(featureVector)
    res = {}
    for i in range(0, 8):
        startingIndex = 16*i
        bandVector = signature[startingIndex: startingIndex + 16]
        bandNumber = int(str(i + 1) + str(generateHash(bandVector)))
        try:
            res[bandNumber].append(key)
        except KeyError:
            res[bandNumber] = key
    return res.items()

# Combine files which are same for at least one band and return list
def combineSimilarFiles(files, fileNames):
    similarFiles = {}
    for file in fileNames:
        if file in files:
#            Remove file from files.
            x = list(files)
            x.remove(file)
            similarFiles[file] = x
            print (similarFiles)
    return similarFiles.items()

# Get SVD for image using V vector that we got from 10 samples
def getMeanSVD(image, vVar, m, std):
    std_svdMat = (image - m) / std
    return np.matmul(std_svdMat,vVar)

# This function will calculate the V matrix for a sample and return the V value. That value is used for other images
def runSvd(partition):
    svdMat = []
    fileNames = []
    number = 0
    for part in partition:
        svdMat.append(part[1])
        fileNames.append(part[0])
        if number > 9:
            break
        number += 1

    m, std = np.mean(svdMat, axis=0), np.std(svdMat, axis=0)
    std = np.where(std == 0,1,std)
    std_svdMat = (svdMat - m) / std
    low_dim = 10
    U, s, V = svd(std_svdMat, full_matrices=0)
    V = V[0:low_dim,:]

    return (V.T, m, std)

if __name__ == "__main__":
    
    config = SparkConf().setAppName("PCA Test")
    sc = SparkContext(conf=config)
    directoryPath = "./sample/*"  # Provide directory path
    outputTxtFile = open("output.txt","w")
    filesRdd = sc.binaryFiles(directoryPath)
    filesRdd.persist()

#    Calculate the feature vector for all images
    featureVectorRDD = filesRdd.flatMap(lambda file: getOrthoTif(file[0], file[1])).mapValues(lambda val: calFeatureVector(val))

    featureVectorRDD.persist()
    fileNames = featureVectorRDD.keys().map(lambda x: x[x.rfind('/') + 1:]).collect()
#    Apply LSH on featureVector
    lshRDD = featureVectorRDD.flatMap(lambda val: applyLSH(val)).groupByKey().mapValues(list)
#    Apply filter to get similar files for 4 given images and combine them
    similarFilesRDD = lshRDD.flatMap(lambda x: combineSimilarFiles(x[1], fileNames)).reduceByKey(lambda a, b: list(set(a + b))).collect()
    similarFilesRDDDict = dict(similarFilesRDD)
    similarFilesList = list(similarFilesRDDDict.keys()) + list(set().union(*similarFilesRDDDict.values()))
#   Taking random 10 samples and applying SVD on it to get V matrix which will be used to tranform all images.
    sample = featureVectorRDD.takeSample(False, 10)
    vVar, m, std = runSvd(sample)
    low_Dimension_Images = featureVectorRDD.mapValues(lambda a: getMeanSVD(a, vVar, m, std)).filter(lambda a: a[0] in similarFilesList).collect()

    low_Dimension_Images = dict(low_Dimension_Images)

    outputTxtFile.write("\n\n\n************  Result  *****************\n\n")
    for key,imageList in similarFilesRDDDict.items():
        distance = dict()
        for img in imageList:
            a = low_Dimension_Images[key]
            b = low_Dimension_Images[img]
            distance[key + " -> " + img] = np.sqrt(np.sum((a-b)**2))
        outputTxtFile.write(str(sorted(distance.items(), key=lambda x: x[1])))

    outputTxtFile.write("\n\n*****************************\n\n\n")


