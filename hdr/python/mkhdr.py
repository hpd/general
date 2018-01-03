#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (c) 2016 Haarm-Pieter Duiker <hpd1@duikerresearch.com>
#

try:
    import cv2
except:
    cv2 = None

import array
import math
import numpy as np
import os
import shutil
import subprocess as sp
import sys
import tempfile
import timeit
import traceback

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ImageBufAlgo, ImageInput, ROI

# Formats with exif data
generalExtensions = ["jpg", "tiff", "tif"]

# Should match
# https://github.com/OpenImageIO/oiio/blob/master/src/raw.imageio/rawinput.cpp#L81
rawExtensions = ["bay", "bmq", "cr2", "crw", "cs1", "dc2", "dcr", "dng",
    "erf", "fff", "hdr", "k25", "kdc", "mdc", "mos", "mrw",
    "nef", "orf", "pef", "pxn", "raf", "raw", "rdc", "sr2",
    "srf", "x3f", "arw", "3fr", "cine", "ia", "kc2", "mef",
    "nrw", "qtk", "rw2", "sti", "rwl", "srw", "drf", "dsc",
    "ptx", "cap", "iiq", "rwz"]

#
# Workaround for OIIO libRaw support
#
temp_dirs = []

def oiioSupportsRaw():
    '''
    Check to see if raw files can be loaded natively
    '''
    # Check to see if the raw plugin has been loaded
    format_list = oiio.get_string_attribute( "format_list" ).split(',')
    raw_plugin_present = 'raw' in format_list

    # Check to see if version is above when raw reading was fixed
    # Update this version number to reflect when the functionality is fixed
    version_great_enough = oiio.VERSION >= 10707

    return (raw_plugin_present and version_great_enough)

def loadImageBuffer( imagePath, outputGamut=None, rawSaturationPoint=-1.0,
    dcrawVariant=None ):
    '''
    Load an image buffer. Manage raw formats if OIIO can't load them directly
    '''
    global temp_dirs

    # Raw camera files require special handling
    imageExtension = os.path.splitext( imagePath )[-1][1:].lower()
    if imageExtension in rawExtensions:

        # Either OIIO can read the data directly
        if oiioSupportsRaw():
            print( "\tUsing OIIO ImageInput to read raw file" )

            # Convert gamut number to text
            gamuts = { 
                0 : "raw", 
                1 : "sRGB",
                2 : "Adobe",
                3 : "Wide",
                4 : "ProPhoto",
                5 : "XYZ"
            }
            outputGamutText = "sRGB"
            if outputGamut in gamuts:
                outputGamutText = gamuts[outputGamut]

            # Spec will be used to configure the file read
            spec = ImageSpec()
            spec.attribute("raw:ColorSpace", outputGamutText)
            spec.attribute("raw:use_camera_wb", 1)
            spec.attribute("raw:auto_bright", 0)
            spec.attribute("raw:use_camera_matrix", 0)
            spec.attribute("raw:adjust_maximum_thr", 0.0)

            imageBuffer = ImageBuf()
            imageBuffer.reset( imagePath, 0, 0, spec )

        # Or we need to use dcraw to help the process along
        else:
            print( "\tUsing dcraw to convert raw, then OIIO to read file" )

            # Create a new temp dir for each image so there's no chance
            # of a name collision
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append( temp_dir )

            imageName = os.path.split(imagePath)[-1]
            temp_file = os.path.join(temp_dir, "%s_temp.tiff" % imageName)

            if outputGamut is None:
                outputGamut = 1

            if dcrawVariant == "dcraw":
                cmd = "dcraw"
                args  = []
                #args += ['-v']
                args += ['-w', '-o', str(outputGamut), '-4', '-T', '-W']
                args += ['-c']
                if rawSaturationPoint > 0.0:
                    args += ['-S', str(int(rawSaturationPoint))]
                args += [imagePath]

                cmdargs = [cmd]
                cmdargs.extend(args)

                #print( "\tTemp_file : %s" % temp_file )
                print( "\tCommand   : %s" % " ".join(cmdargs) )

                with open(temp_file, "w") as temp_handle:
                    process = sp.Popen(cmdargs, stdout=temp_handle, stderr=sp.STDOUT)
                    process.wait()

            # Use the libraw dcraw_emu when dcraw doesn't support a camera yet
            else:
                cmd = "dcraw_emu"
                args  = []
                args += ['-w', '-o', str(outputGamut), '-4', '-T', '-W']

                #if rawSaturationPoint > 0.0:
                #    args += ['-c', str(float(rawSaturationPoint/16384.0))]
                if rawSaturationPoint > 0.0:
                    args += ['-S', str(int(rawSaturationPoint))]
                args += [imagePath]

                cmdargs = [cmd]
                cmdargs.extend(args)

                print( "\tCommand   : %s" % " ".join(cmdargs) )

                dcraw_emu_temp_file = "%s.tiff" % imageName
                process = sp.Popen(cmdargs, stderr=sp.STDOUT)
                process.wait()

                print( "\tMoving temp file to : %s" % temp_dir )
                shutil.move( dcraw_emu_temp_file, temp_file )

            #print( "Loading   : %s" % temp_file )
            imageBuffer = ImageBuf( temp_file )

    # Just load the image using OIIO
    else:
        #print( "Using OIIO ImageBuf read route" )
        imageBuffer = ImageBuf( imagePath )

    return imageBuffer


#
# Use OIIO ImageBuf processing
#
def ImageAttributes(inputImage):
    '''
    Get image bit channel type, width, height, number of channels and metadata
    '''
    inputImageSpec = inputImage.spec()
    channelType = inputImageSpec.format.basetype
    orientation = inputImage.orientation
    width = inputImageSpec.width
    height = inputImageSpec.height

    channels = inputImageSpec.nchannels
    metadata = inputImageSpec.extra_attribs
    return (channelType, width, height, channels, orientation, metadata, inputImageSpec)

def ImageBufMakeConstant(xres, 
    yres, 
    chans=3, 
    format=oiio.UINT8, 
    value=(0,0,0),
    xoffset=0, 
    yoffset=0,
    orientation=1,
    inputSpec=None) :
    '''
    Create a new Image Buffer
    '''
    
    # Copy an existing input spec
    # Mostly to ensure that metadata makes it through
    if inputSpec:
        spec = inputSpec
        spec.width = xres
        spec.height = yres
        spec.nchannels = chans
        spec.set_format( format )

    # Or create a new ImageSpec
    else:
        spec = ImageSpec (xres,yres,chans,format)

    spec.x = xoffset
    spec.y = yoffset
    b = ImageBuf (spec)
    b.orientation = orientation
    oiio.ImageBufAlgo.fill(b, value)

    return b

def ImageBufWrite(imageBuf, 
    filename, 
    format=oiio.UNKNOWN,
    compression=None,
    compressionQuality=0,
    metadata=None,
    additionalAttributes=None):
    '''
    Write an Image Buffer
    '''
    outputSpec = imageBuf.specmod()
    if compression:
        outputSpec.attribute("compression", compression)
            
        if compressionQuality > 0:
            outputSpec.attribute("CompressionQuality", compressionQuality)

    if metadata:
        for attr in metadata:
            outputSpec.attribute(attr.name, attr.value)

    if additionalAttributes:
        for key, value in additionalAttributes.iteritems():
            outputSpec.attribute("mkhdr:%s" % key, str(value))

    if not imageBuf.has_error:
        imageBuf.set_write_format( format )
        imageBuf.write( filename )
    if imageBuf.has_error:
        print( "Error writing", filename, ":", imageBuf.geterror() )
        return False

    return True

def ImageBufReorient(imageBuf, orientation):
    '''
    Resets the orientation of the image
    '''

    '''
    Orientation 6 and 8 seem to be reversed in OIIO, at least for Canon
    cameras... This needs to be investigated further.
    '''
    if orientation == 6:
        imageBuf.specmod().attribute ("Orientation", 1)
        ImageBufAlgo.rotate270(imageBuf, imageBuf)
        ImageBufAlgo.reorient (imageBuf, imageBuf)

    elif orientation == 8:
        imageBuf.specmod().attribute ("Orientation", 1)
        ImageBufAlgo.rotate90(imageBuf, imageBuf)
        ImageBufAlgo.reorient (imageBuf, imageBuf)

    else:
        ImageBufAlgo.reorient (imageBuf, imageBuf)

def ImageBufWeight(weight, inputBuffer, gamma=0.75, clip=0.05, lut=None):
    '''
    Apply a bell / triangular weight function to an Image Buffer
    '''
    (channelType, width, height, channels, orientation, metadata, inputSpec) = ImageAttributes(inputBuffer)
    
    temp     = ImageBufMakeConstant(width, height, channels, oiio.HALF )
    grey05   = ImageBufMakeConstant(width, height, channels, oiio.HALF, tuple([0.5]*channels) )
    
    if lut:
        ImageBufAlgo.add(temp, temp, inputBuffer)
        if 1 in lut:
            ImageBufAlgo.clamp(temp, temp, tuple([0.5]*channels), tuple([1.0]*channels))

        if 2 in lut:
            ImageBufAlgo.clamp(temp, temp, tuple([0.0]*channels), tuple([0.5]*channels))

        #print( "\tLUT application : %s" % result )
        ImageBufAlgo.absdiff(temp, grey05, temp)
    else:
        ImageBufAlgo.absdiff(temp, grey05, inputBuffer)
    
    ImageBufAlgo.sub(temp, grey05, temp)
    ImageBufAlgo.div(temp, temp, 0.5)

    ImageBufAlgo.sub(temp, temp, clip)
    ImageBufAlgo.mul(temp, temp, 1.0/(1.0-clip))

    ImageBufAlgo.clamp(temp, temp, tuple([0.0]*channels), tuple([1.0]*channels))
    ImageBufAlgo.pow(weight, temp, gamma)

def findAverageWeight(imageBuffer, width, height, channels):
    '''
    Get the average value of the weighted image
    '''
    weight    = ImageBufMakeConstant(width, height, channels, oiio.HALF)
    temp      = ImageBufMakeConstant(1, 1, channels, oiio.HALF)

    print( "\tComputing Weight" )
    ImageBufWeight(weight, imageBuffer)
    # Compute the average weight by resizing to 1x1
    print( "\tResizing" )
    # The nthreads argument doesn't seem to have much effect
    ImageBufAlgo.resize(temp, weight, nthreads=cpu_count(), filtername='box')
    # Get the average weight value
    averageWeight = sum(map(float, temp.getpixel(0,0)))/channels

    return averageWeight

def findBaseExposureIndexSerial(imageBuffers, width, height, channels):
    '''
    Find the base exposure out of series of Image Buffers
    Images are processed serially
    '''
    t0 = timeit.default_timer()

    print( "findBaseExposureIndex - Serial" )

    highestWeightIndex = 0
    highestWeight = 0
    for i in range(len(imageBuffers)):
        print( "Exposure : %d" % i )
        # Compute the average pixel weight
        averageWeight = findAverageWeight(imageBuffers[i], width, height, channels)
        print( "\tAverage weight : %s" % averageWeight )
        if averageWeight > highestWeight:
            highestWeight = averageWeight
            highestWeightIndex = i
            print( "New highest weight index : %s" % highestWeightIndex )

    print( "Base Exposure Index : %d" % highestWeightIndex )

    t1 = timeit.default_timer()
    elapsed = t1 - t0
    
    print( "Finding base exposure index took %s seconds" % (elapsed) )

    return highestWeightIndex

from multiprocessing import Pool, Lock, cpu_count

def findAverageWeightFromPath(inputPath, width, height, channels):
    '''
    Find the average weight of an image, specified by it's file path
    '''
    weight    = ImageBufMakeConstant(width, height, channels, oiio.HALF)
    temp      = ImageBufMakeConstant(1, 1, channels, oiio.HALF)

    try:
        print( "\tReading image : %s" % inputPath )
        inputBufferRaw = ImageBuf( inputPath )

        # Cast to half by adding with a const half buffer.
        inputBufferHalf = ImageBufMakeConstant(width, height, channels, oiio.HALF)
        ImageBufAlgo.add(inputBufferHalf, inputBufferHalf, inputBufferRaw)

        print( "\tComputing Weight" )
        ImageBufWeight(weight, inputBufferHalf)
        # Compute the average weight by resizing to 1x1
        print( "\tResizing" )
        # Not using multithreading here, as this function will be called within
        # Python's multhreading framework
        ImageBufAlgo.resize(temp, weight, filtername='box')
        # Get the average weight value
        weight = temp.getpixel(0,0)
        #print( "\tWeight : %s" % str(weight) )

        averageWeight = sum(map(float, weight))/channels
    except Exception, e:
        print( "Exception in findAverageWeightFromPath" )
        print( repr(e) )

    return averageWeight

def findAverageWeightFromPath_splitargs(args):
    '''
    findAverageWeight_splitargs splits the single argument 'args' into mulitple
    arguments. This is needed because map() can only be used for functions
    that take a single argument (see http://stackoverflow.com/q/5442910/1461210)
    '''

    try:
        return findAverageWeightFromPath(*args)
    except Exception, e:
        pass

def findBaseExposureIndexMultithreaded(inputPaths, width, height, channels, multithreaded):
    '''
    Find the base exposure out of series of image paths
    Images are processed in parallel
    '''
    t0 = timeit.default_timer()

    print( "findBaseExposureIndex - Multithreaded (%d threads)" % multithreaded )

    try:
        pool = Pool(processes=multithreaded)

        result = pool.map_async(findAverageWeightFromPath_splitargs,
            [(inputPath, 
                width, 
                height,
                channels) for inputPath in inputPaths],
            chunksize=1)
        try:
            averageWeights = result.get(0xFFFF)
        except KeyboardInterrupt:
            print( "\nProcess received Ctrl-C. Exiting.\n" )
            return
        except:
            print( "\nCaught exception. Exiting." )
            print( '-'*60 )
            traceback.print_exc()
            print( '-'*60 )
            return
    except:
        print( "Error in multithreaded processing. Exiting." )
        print( '-'*60 )
        traceback.print_exc()
        print( '-'*60 )

    for i in range(len(inputPaths)):
        print( "Image %d - Weight : %s" % (i, averageWeights[i]))

    highestWeightIndex = averageWeights.index(max(averageWeights))
    print( "Base Exposure Index : %d" % highestWeightIndex )

    t1 = timeit.default_timer()
    elapsed = t1 - t0
    
    print( "Finding base exposure index took %s seconds" % (elapsed) )

    return highestWeightIndex

def OpenCVImageBufferFromOIIOImageBuffer(oiioImageBuffer):
    oiioSpec = oiioImageBuffer.spec()
    (width, height, channels) = (oiioSpec.width, oiioSpec.height, oiioSpec.nchannels)
    oiioFormat = oiioSpec.format
    oiioChanneltype = oiioFormat.basetype

    # Promote halfs to full float as Python may not handle those properly
    if oiioChanneltype == oiio.BASETYPE.HALF:
        oiioChanneltype = oiio.BASETYPE.FLOAT

    oiioToNPBitDepth = {
        oiio.BASETYPE.UINT8  : np.uint8,
        oiio.BASETYPE.UINT16 : np.uint16,
        oiio.BASETYPE.UINT32 : np.uint32,
        oiio.BASETYPE.HALF   : np.float16,
        oiio.BASETYPE.FLOAT  : np.float32,
        oiio.BASETYPE.DOUBLE : np.float64,
    }

    # Default to float
    if oiioChanneltype in oiioToNPBitDepth:
        npChannelType = oiioToNPBitDepth[oiioChanneltype]
    else:
        print( "oiio to opencv - Using fallback bit depth" )
        npChannelType = np.float32

    opencvImageBuffer = np.array(oiioImageBuffer.get_pixels(oiioChanneltype), dtype=npChannelType).reshape(height, width, channels)

    return opencvImageBuffer

def OIIOImageBufferFromOpenCVImageBuffer(opencvImageBuffer):
    (height, width, channels) = opencvImageBuffer.shape
    npChanneltype = opencvImageBuffer.dtype

    npToArrayBitDepth = {
        np.dtype('uint8')   : 'B',
        np.dtype('uint16')  : 'H',
        np.dtype('uint32')  : 'I',
        np.dtype('float32') : 'f',
        np.dtype('float64') : 'd',
    }

    npToOIIOBitDepth = {
        np.dtype('uint8')   : oiio.BASETYPE.UINT8,
        np.dtype('uint16')  : oiio.BASETYPE.UINT16,
        np.dtype('uint32')  : oiio.BASETYPE.UINT32,
        np.dtype('float32') : oiio.BASETYPE.FLOAT,
        np.dtype('float64') : oiio.BASETYPE.DOUBLE,
    }

    # Support this when oiio more directly integrates with numpy
    #    np.dtype('float16') : oiio.BASETYPE.HALF,

    if (npChanneltype in npToArrayBitDepth and 
        npChanneltype in npToOIIOBitDepth):
        arrayChannelType = npToArrayBitDepth[npChanneltype]
        oiioChanneltype = npToOIIOBitDepth[npChanneltype]
    else:
        print( "opencv to oiio - Using fallback bit depth" )
        arrayChannelType = 'f'
        oiioChanneltype = oiio.BASETYPE.FLOAT

    spec = ImageSpec(width, height, channels, oiioChanneltype)
    oiioImageBuffer = ImageBuf(spec)
    roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
    conversion = oiioImageBuffer.set_pixels( roi, array.array(arrayChannelType, opencvImageBuffer.flatten()) )
    if not conversion:
        print( "opencv to oiio - Error converting the OpenCV buffer to an OpenImageIO buffer" )
        oiioImageBuffer = None

    return oiioImageBuffer

def find2dAlignmentMatrix(im1, im2, 
    warp_mode = cv2.MOTION_TRANSLATION, 
    center_crop_resolution=0,
    brightness_scale=1.0):
    '''
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_mode = cv2.MOTION_AFFINE
    warp_mode = cv2.MOTION_HOMOGRAPHY
    '''

    #width, height, channels,

    warpModeToText = { 
        0 : 'Translation',
        1 : 'Euclidean',
        2 : 'Affine', 
        3 : 'Homography'
    }

    scale_factor = 1.0

    # Convert to from OIIO to OpenCV-friendly format
    res1 = OpenCVImageBufferFromOIIOImageBuffer(im1)
    res2 = OpenCVImageBufferFromOIIOImageBuffer(im2)

    (height, width, channels) = res1.shape

    if center_crop_resolution >= 0:
        if center_crop_resolution == 0:
            center_crop_resolution = min(width, height)/2

        if width > center_crop_resolution:
            print( width, height )
            ws = width/2 - center_crop_resolution/2
            we = width/2 + center_crop_resolution/2
            hs = height/2 - center_crop_resolution/2
            he = height/2 + center_crop_resolution/2

            print( "Center crop : %d-%d, %d-%d" % (ws, we, hs, he) )
            print( "Cropping image 1")
            res1 = res1[hs:he, ws:we]
            print( "Cropping image 2")
            res2 = res2[hs:he, ws:we]
            scale_factor = 1.0

    #print( res1.shape )
    #print( res2.shape )

    # Scale image1 - Not entirely necessary
    res1 = cv2.multiply(res1, np.array([brightness_scale]))
    
    # Convert images to grayscale
    #if len(im1shape) > 2:
    if channels > 1:
        im1_gray = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = res1
        im2_gray = res2

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    print( "Aligning - Mode : %s, %d" % (warpModeToText[warp_mode], warp_mode) )

    #cv2.imwrite("im1_gray.exr", im1_gray)
    #cv2.imwrite("im2_gray.exr", im2_gray)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
    except Exception, e:
        print( "Exception in findTransformECC : %s" % repr(e))
        warp_matrix = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    w = len(warp_matrix[0])
    h = len(warp_matrix)
    print( "Alignment Matrix - %d x %d" % (w, h) )
    for j in range(h):
        print( map(lambda x: "%3.6f" % x, warp_matrix[j]) )
    #print( "Alignment Matrix : %s" % warp_matrix )
    print( "Translation" )
    for j in range(h):
        print( "%3.6f" % (float(warp_matrix[j][-1])*scale_factor) )

    # OpenCV warp
    '''
    print( "OpenCV warp" )
    sz = im1_gray.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(res2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    output = "test.exr"
    print( "Writing output : %s" % output )
    cv2.imwrite(output, im2_aligned)
    '''

    return warp_matrix

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)

def mkhdr(outputPath, 
    inputPaths, 
    responseLUTPaths, 
    baseExposureIndex, 
    writeIntermediate = False, 
    outputGamut = 1,
    compression = None,
    compressionQuality = 0,
    rawSaturationPoint = -1.0,
    alignImages = False,
    dcrawVariant = None):
    '''
    Create an HDR image from a series of individual exposures
    If the images are non-linear, a series of response LUTs can be used to
    linearize the data
    '''

    global temp_dirs

    # Set up capture of 
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_stdout = sys.stdout = Logger()
    redirected_stderr = sys.stderr = Logger()

    # Create buffers for inputs
    inputBuffers = []
    inputAttributes = []

    # Read images
    for inputPath in inputPaths:
        print( "Reading input image : %s" % inputPath )
        # Read
        inputBufferRaw = loadImageBuffer( inputPath, outputGamut=outputGamut, 
            rawSaturationPoint=rawSaturationPoint,
            dcrawVariant=dcrawVariant )

        # Reset the orientation
        print( "\tRaw Orientation : %d" % inputBufferRaw.orientation)
        ImageBufReorient(inputBufferRaw, inputBufferRaw.orientation)

        # Get attributes
        (channelType, width, height, channels, orientation, metadata, inputSpec) = ImageAttributes(inputBufferRaw)

        # Cast to half by adding with a const half buffer.
        inputBufferHalf = ImageBufMakeConstant(width, height, channels, oiio.HALF)
        ImageBufAlgo.add(inputBufferHalf, inputBufferHalf, inputBufferRaw)

        # Get exposure-specific information
        exposure = getExposureInformation(metadata)

        print( "\tChannel Type : %s" % (channelType) )
        print( "\tWidth        : %s" % (width) )
        print( "\tHeight       : %s" % (height) )
        print( "\tChannels     : %s" % (channels) )
        print( "\tOrientation  : %s" % (orientation) )
        print( "\tExposure     : %s" % (exposure) )
        print( "\tMetadata #   : %s" % (len(metadata)) )

        # Store pixels and image attributes
        inputBuffers.append( inputBufferHalf )
        inputAttributes.append( (channelType, width, height, channels, orientation, metadata, exposure, inputSpec) )

    # Get the base exposure information
    # All other images will be scaled to match this exposure
    if baseExposureIndex >= 0:
        baseExposureIndex = max(0, min(len(inputPaths)-1, baseExposureIndex))
    else:
        multithreaded = True
        if multithreaded:
            threads = cpu_count()
            baseExposureIndex = findBaseExposureIndexMultithreaded(inputPaths, width, height, channels, threads)
        else:
            baseExposureIndex = findBaseExposureIndexSerial(inputBuffers, width, height, channels)

    baseExposureMetadata = inputAttributes[baseExposureIndex][5]
    baseExposureInfo = inputAttributes[baseExposureIndex][6]
    baseInputspec = inputAttributes[baseExposureIndex][7]

    print( "" )
    print( "Base exposure index : %d" % baseExposureIndex )
    print( "Base exposure info  : %s" % baseExposureInfo )

    # Find the lowest and highest exposures
    exposureAdjustments = [getExposureAdjustment(x[6], baseExposureInfo) for x in inputAttributes]

    minExposureOffsetIndex = exposureAdjustments.index(min(exposureAdjustments))
    maxExposureOffsetIndex = exposureAdjustments.index(max(exposureAdjustments))

    print( "Max exposure index  : %d" % minExposureOffsetIndex )
    print( "Min exposure index  : %d" % maxExposureOffsetIndex )

    print( "\nBegin processing\n" )

    # Two buffers needed for algorithm
    imageSum  = ImageBufMakeConstant(width, height, channels, oiio.HALF, 
        inputSpec=baseInputspec)
    weightSum = ImageBufMakeConstant(width, height, channels, oiio.HALF)

    # Re-used intermediate buffers
    color     = ImageBufMakeConstant(width, height, channels, oiio.HALF)
    weight    = ImageBufMakeConstant(width, height, channels, oiio.HALF)
    weightedColor = ImageBufMakeConstant(width, height, channels, oiio.HALF)

    # Process images
    for inputIndex in range(len(inputPaths)):
        inputPathComponents = (os.path.splitext( inputPaths[inputIndex] )[0], ".exr")
        intermediate = 0

        ImageBufAlgo.zero( color )
        ImageBufAlgo.zero( weight )
        ImageBufAlgo.zero( weightedColor )

        print( "Processing input image : %s" % inputPaths[inputIndex] )
        inputBuffer = inputBuffers[inputIndex]

        # Copy the input buffer data
        ImageBufAlgo.add(color, color, inputBuffer)

        if writeIntermediate:
            intermediatePath = "%s_int%d.float_buffer%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(color, intermediatePath)

        # Find the image alignment matrix to align this exposure with the base exposure
        if alignImages:
            try:
                if inputIndex != baseExposureIndex:
                    if cv2:
                        print( "\tAligning image %d to base exposure %d " % (inputIndex, baseExposureIndex) )
                        warpMatrix = find2dAlignmentMatrix(inputBuffer, inputBuffers[baseExposureIndex])

                        # reformat for OIIO's warp
                        w = map(float, list(warpMatrix.reshape(1,-1)[0]))
                        warpTuple = (w[0], w[1], 0.0, w[3], w[4], 0.0, w[2], w[5], 1.0)
                        print( warpTuple )

                        warped = ImageBuf()
                        result = ImageBufAlgo.warp(warped, color, warpTuple)
                        if result:
                            print( "\tImage alignment warp succeeded." )
                            if writeIntermediate:
                                intermediatePath = "%s_int%d.warped%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
                                intermediate += 1
                                ImageBufWrite(warped, intermediatePath)

                            color = warped
                        else:
                            print( "\tImage alignment warp failed." )
                            if writeIntermediate:
                                intermediate += 1
                    else:
                        print( "\tSkipping image alignment. OpenCV not defined" )
                        if writeIntermediate:
                            intermediate += 1
                else:
                    print( "\tSkipping alignment of base exposure to itself")
                    if writeIntermediate:
                        intermediate += 1

            except:
                print( "Exception in image alignment" )
                print( '-'*60 )
                traceback.print_exc()
                print( '-'*60 )

        # Weight
        print( "\tComputing image weight" )

        lut = []
        if inputIndex == minExposureOffsetIndex:
            lut.append(1)
        if inputIndex == maxExposureOffsetIndex:
            lut.append(2)
        if lut:
            print( "\tUsing LUT %s in weighting calculation" % lut )
        ImageBufWeight(weight, color, lut=lut)

        if writeIntermediate:
            intermediatePath = "%s_int%d.weight%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(weight, intermediatePath)

        # Linearize using LUTs
        if responseLUTPaths:
            for responseLUTPath in responseLUTPaths:
                print( "\tApplying LUT %s" % responseLUTPath )
                ImageBufAlgo.ociofiletransform(color, color, os.path.abspath(responseLUTPath) )

                if writeIntermediate:
                    intermediatePath = "%s_int%d.linearized%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
                    intermediate += 1
                    ImageBufWrite(color, intermediatePath)

        # Get exposure offset
        inputExposureInfo = inputAttributes[inputIndex][6]
        exposureAdjustment = getExposureAdjustment(inputExposureInfo, baseExposureInfo)
        exposureScale = pow(2, exposureAdjustment)

        # Re-expose input
        print( "\tScaling by %s stops (%s mul)" % (exposureAdjustment, exposureScale) )
        ImageBufAlgo.mul(color, color, exposureScale)

        if writeIntermediate:
            intermediatePath = "%s_int%d.exposure_adjust%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(color, intermediatePath)

        # Multiply color by weight
        print( "\tMultiply by weight" )

        ImageBufAlgo.mul(weightedColor, weight, color)

        if writeIntermediate:
            intermediatePath = "%s_int%d.color_x_weight%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(weightedColor, intermediatePath)

        print( "\tAdd values into sum" )

        # Sum weighted color and weight
        ImageBufAlgo.add(imageSum,  imageSum,  weightedColor)
        ImageBufAlgo.add(weightSum, weightSum, weight)

        if writeIntermediate:
            intermediatePath = "%s_int%d.color_x_weight_sum%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(imageSum, intermediatePath)

            intermediatePath = "%s_int%d.weight_sum%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(weightSum, intermediatePath)

    # Divid out weights
    print( "Dividing out weights" )
    ImageBufAlgo.div(imageSum, imageSum, weightSum)

    # Write to disk
    print( "Writing result : %s" % outputPath )

    # Restore regular streams
    sys.stdout, sys.stderr = old_stdout, old_stderr

    additionalAttributes = {}
    additionalAttributes['inputPaths'] = " ".join(inputPaths)
    additionalAttributes['stdout'] = "".join(redirected_stdout.log)
    additionalAttributes['stderr'] = "".join(redirected_stderr.log)

    ImageBufWrite(imageSum, outputPath, 
        compression=compression,
        compressionQuality=compressionQuality,
        metadata=baseExposureMetadata,
        additionalAttributes=additionalAttributes)

    # Clean up temp folders
    for temp_dir in temp_dirs:
        #print( "Removing : %s" % temp_dir )
        shutil.rmtree(temp_dir)

    for temp_dir in temp_dirs:
        temp_dirs.remove(temp_dir)


#
# Exposure information hdr generation
#
def getExposureValue(exposureInfo):
    shutter = exposureInfo['ExposureTime']
    aperture = exposureInfo['FNumber']
    iso = 100.0
    if 'Exif:PhotographicSensitivity' in exposureInfo:
        iso = float(exposureInfo['Exif:PhotographicSensitivity'])
    elif 'Exif:ISOSpeedRatings' in exposureInfo:
        iso = float(exposureInfo['Exif:ISOSpeedRatings']*100.0)

    #bias = exposureInfo['Exif:ExposureBiasValue']
    #print( "getExposureValue - %3.6f - %3.6f - %3.6f - %3.6f" % (shutter, aperture, iso, bias) )

    ev = math.log((100.0*aperture*aperture)/(iso * shutter))/math.log(2.0)

    return ev

def getExposureAdjustment(exposureInfo, baseExposureInfo):
    evBase = getExposureValue(baseExposureInfo)
    ev = getExposureValue(exposureInfo)
    
    #print( "getExposureAdjustment - %3.6f - %3.6f = %3.6f" % (evBase, ev, (ev-evBase)) )
    
    return (ev - evBase)

def getExposureInformation(metadata):
    exposure = {}
    for attr in metadata:
        #print( "\t%s : %s" % (attr.name, attr.value) )
        if attr.name in ['ExposureTime', 
            'FNumber',
            'Exif:PhotographicSensitivity', 
            'Exif:ISOSpeedRatings',
            'Exif:ApertureValue',
            'Exif:BrightnessValue', 
            'Exif:ExposureBiasValue']:
            #print( "\tStoring %s : %s" % (attr.name, attr.value) )
            exposure[attr.name] = attr.value

    return exposure

def mergeHDRGroup(imageUris,
    writeIntermediate = False,
    responseLUTPath = None,
    baseExposureIndex = None,
    outputPath = None,
    outputGamut = 1,
    compression = None,
    compressionQuality = 0,
    rawSaturationPoint = -1.0,
    alignImages = False,
    dcrawVariant = None):
    print( "" )
    print( "Merge images into an HDR - begin" )
    print( "" )
    print( "Images : %s" % imageUris )

    # Set up other resources
    if baseExposureIndex is None:
        baseExposureIndex = -1
    else:
        print( "Base Exposure : %s" % baseExposureIndex )

    if os.path.isdir( outputPath ):
        outputPath = "%s/%s%s" % (outputPath, os.path.splitext( imageUris[baseExposureIndex] )[0], ".exr")
    elif not outputPath:
        outputPath = "%s%s" % (os.path.splitext( imageUris[baseExposureIndex] )[0], ".exr")

    luts = []
    if responseLUTPath:
        luts = [responseLUTPath]

    # Merge the HDR images
    mkhdr(outputPath, 
        imageUris, 
        luts, 
        baseExposureIndex, 
        writeIntermediate,
        outputGamut = outputGamut,
        compression = compression,
        compressionQuality = compressionQuality,
        rawSaturationPoint = rawSaturationPoint,
        alignImages = alignImages,
        dcrawVariant = dcrawVariant)

    print( "" )
    print( "Merge images into an HDR - end" )
    print( "" )

def mergeHDRFolder(hdrDir, 
    responseLUTPath = None,
    writeIntermediate = False, 
    baseExposureIndex = None,
    outputPath = None,
    outputGamut = 1,
    compression = None,
    compressionQuality = 0,
    rawSaturationPoint = -1.0,
    alignImages = False,
    dcrawVariant = None):
    startingDir = os.getcwd()

    print( "mergeHDRFolder - folder : %s" % hdrDir )

    try:
        extensions = generalExtensions
        extensions.extend( rawExtensions )

        imageUris = sorted( os.listdir( hdrDir ) )
        imageUris = [x for x in imageUris if os.path.splitext(x)[-1].lower()[1:] in extensions]
        imageUris = [x for x in imageUris if x[0] != '.']

        print( "mergeHDRFolder - images : %s" % imageUris )

        os.chdir( hdrDir )
        mergeHDRGroup( imageUris, 
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate,
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath,
            outputGamut = outputGamut,
            compression = compression,
            compressionQuality = compressionQuality,
            rawSaturationPoint = rawSaturationPoint,
            alignImages = alignImages,
            dcrawVariant = dcrawVariant)
    except Exception, e:
        print( "Exception in HDR merging" )
        print( '-'*60 )
        traceback.print_exc()
        print( '-'*60 )

    os.chdir( startingDir )

def mergeHDRFolderMulti(hdrDir, 
    bracketSize = 3,
    responseLUTPath = None,
    writeIntermediate = False, 
    baseExposureIndex = None,
    outputPath = None,
    outputGamut = 1,
    compression = None,
    compressionQuality = 0,
    rawSaturationPoint = -1.0,
    alignImages = False,
    dcrawVariant = None):
    startingDir = os.getcwd()

    print( "mergeHDRFolderMulti - folder : %s" % hdrDir )

    try:
        extensions = generalExtensions
        extensions.extend( rawExtensions )

        multiImageUris = sorted( os.listdir( str(hdrDir) ) )
        multiImageUris = [x for x in multiImageUris if os.path.splitext(x)[-1].lower()[1:] in extensions]
        multiImageUris = [x for x in multiImageUris if x[0] != '.']

        bracketedGroups = len( multiImageUris )/bracketSize
        for group in range( bracketedGroups ):
            imageUris = multiImageUris[group*bracketSize:(group+1)*bracketSize]

            print( "mergeHDRFolderMulti %d - images : %s" % (group, imageUris) )

            os.chdir( hdrDir )
            mergeHDRGroup( imageUris, 
                responseLUTPath = responseLUTPath,
                writeIntermediate = writeIntermediate,
                baseExposureIndex = baseExposureIndex,
                outputPath = outputPath,
                outputGamut = outputGamut,
                compression = compression,
                compressionQuality = compressionQuality,
                rawSaturationPoint = rawSaturationPoint,
                alignImages = alignImages,
                dcrawVariant = dcrawVariant)

    except Exception, e:
        print( "Exception in HDR merging" )
        print( '-'*60 )
        traceback.print_exc()
        print( '-'*60 )

    os.chdir( startingDir )


#
# Get the options, load a set of images and merge them
#
def main():
    import optparse

    usage  = "%prog [options]\n"
    usage += "\n"
    usage += "compression options:\n"
    usage += " exr format compression options  : none, rle, zip, zips(default), piz, pxr24, b44, b44a, dwaa, or dwab\n"
    usage += "   dwaa and dwab compression support depends on the version of OpenImageIO that you're using.\n"
    usage += " tiff format compression options : none, lzw, zip(default), packbits\n"
    usage += " tga format compression options  : none, rle\n"
    usage += " sgi format compression options  : none, rle\n"
    usage += "\n"
    usage += "compression quality options:\n"
    usage += " jpg format compression quality options  : 0 to 100\n"

    p = optparse.OptionParser(description='Merge a set of LDR images into a single HDR result',
                                prog='mkhdr',
                                version='mkhdr',
                                usage=usage)

    p.add_option('--input', '-i', type='string', action='append')
    p.add_option('--inputFolder', default=None)
    p.add_option('--multiInputFolder', default=None)
    p.add_option('--bracketSize', default=3, type='int')
    p.add_option('--output', '-o', default=None)
    p.add_option('--responseLUT', '-r', default=None)
    p.add_option('--baseExposure', '-b', default=-1, type='int')
    p.add_option('--writeIntermediate', '-w', action="store_true")
    p.add_option('--verbose', '-v', action="store_true")
    p.add_option('--gamut', '-g', default=1, type='int',
        help="[0-5], Default 1, Output gamut : raw, sRGB, Adobe, Wide, ProPhoto, XYZ")
    p.add_option("--compression", type='string')
    p.add_option("--quality", type="int", dest="quality", default = -1)
    p.add_option("--rawSaturationPoint", '-s', type="float", default = -1.0)
    p.add_option('--alignImages', '-a', action="store_true")
    p.add_option('--dcraw', default='dcraw_emu', type='string',
        help="dcraw, dcraw_emu")

    options, arguments = p.parse_args()

    #
    # Get options
    # 
    inputPaths = options.input
    inputFolder = options.inputFolder
    multiInputFolder = options.multiInputFolder
    bracketSize = options.bracketSize
    outputPath = options.output
    responseLUTPath = options.responseLUT
    writeIntermediate = options.writeIntermediate == True
    verbose = options.verbose == True
    baseExposureIndex = options.baseExposure
    gamut = options.gamut
    compression = options.compression
    compressionQuality = options.quality
    rawSaturationPoint = options.rawSaturationPoint
    alignImages = options.alignImages == True
    dcrawVariant = options.dcraw

    try:
        argsStart = sys.argv.index('--') + 1
        args = sys.argv[argsStart:]
    except:
        argsStart = len(sys.argv)+1
        args = []

    if verbose:
        print( "command line : \n%s\n" % " ".join(sys.argv) )

    # Process a folder 
    if multiInputFolder:
        mergeHDRFolderMulti(multiInputFolder, 
            bracketSize = bracketSize,
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate, 
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath,
            outputGamut = gamut,
            compression = compression,
            compressionQuality = compressionQuality,
            rawSaturationPoint = rawSaturationPoint,
            alignImages = alignImages,
            dcrawVariant = dcrawVariant)

    elif inputFolder:
        mergeHDRFolder(inputFolder, 
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate, 
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath,
            outputGamut = gamut,
            compression = compression,
            compressionQuality = compressionQuality,
            rawSaturationPoint = rawSaturationPoint,
            alignImages = alignImages,
            dcrawVariant = dcrawVariant)

    # Process an explicit list of files
    elif inputPaths:
        mergeHDRGroup(inputPaths,
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate,
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath,
            outputGamut = gamut,
            compression = compression,
            compressionQuality = compressionQuality,
            rawSaturationPoint = rawSaturationPoint,
            alignImages = alignImages,
            dcrawVariant = dcrawVariant)
    # Missing one type of proper input
    else:
        p.print_help()

# main

if __name__ == '__main__':
    main()
