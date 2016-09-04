#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import sys
import timeit
import traceback

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ImageBufAlgo

#
# LUT related functions
#
def loadResponse( response ):
    '''
    Extremely simple / fragile reader for the Sony spi1d format
    '''
    with open(response) as f:
         content = f.readlines()
         content = [x.strip('\n').strip() for x in content]
    
    entries = len(content)-6
    channels = len(content[6].split())
    
    #print( "entries : %d, channels : %d" % (entries, channels) )
    
    lut = np.zeros((channels, entries), dtype=np.float32)    
    
    for i in range(5,len(content)-1):
        values = content[i].split()
        for c in range(channels):
            lut[c][i-5] = float(values[c])
    
    return lut

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

def ImageBufWrite(imageBuf, filename, format=oiio.UNKNOWN):
    '''
    Write an Image Buffer
    '''
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

def mkhdr(outputPath, inputPaths, responseLUTPaths, baseExposureIndex, writeIntermediate=False):
    '''
    Create an HDR image from a series of individual exposures
    If the images are non-linear, a series of response LUTs can be used to
    linearize the data
    '''

    # Create buffers for inputs
    inputBuffers = []
    inputAttributes = []

    # Read images
    for inputPath in inputPaths:
        print( "Reading input image : %s" % inputPath )
        # Read
        inputBufferRaw = ImageBuf( inputPath )

        # Reset the orientation
        ImageBufReorient(inputBufferRaw, inputBufferRaw.orientation)

        # Get attributes
        (channelType, width, height, channels, orientation, metadata, inputSpec) = ImageAttributes(inputBufferRaw)

        # Cast to half by adding with a const half buffer.
        inputBufferHalf = ImageBufMakeConstant(width, height, channels, oiio.HALF)
        ImageBufAlgo.add(inputBufferHalf, inputBufferHalf, inputBufferRaw)

        # Get exposure-specific information
        exposure = getExposureInformation(metadata)

        print( "\tAttributes : %s, %s, %s, %s, %s, %s" % (channelType, width, height, channels, orientation, len(metadata)) )
        print( "\tExposure   : %s" % (exposure) )

        # Store pixels and image attributes
        inputBuffers.append( inputBufferHalf )
        inputAttributes.append( (channelType, width, height, channels, orientation, metadata, exposure, inputSpec) )

    # Get the base exposure information
    # All other images will be scaled to match this exposure
    print( "Base Exposure Index : %d" % baseExposureIndex )
    if baseExposureIndex >= 0:
        baseExposureIndex = max(0, min(len(inputPaths)-1, baseExposureIndex))
    else:
        multithreaded = True
        if multithreaded:
            threads = cpu_count()
            baseExposureIndex = findBaseExposureIndexMultithreaded(inputPaths, width, height, channels, threads)
        else:
            baseExposureIndex = findBaseExposureIndexSerial(inputBuffers, width, height, channels)

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

        print( "\tScaling by %s stops (%s mul)" % (exposureAdjustment, exposureScale) )

        # Re-expose input
        ImageBufAlgo.mul(color, color, exposureScale)

        if writeIntermediate:
            intermediatePath = "%s_int%d.exposure_adjust%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(color, intermediatePath)

        print( "\tComputing image weight" )

        # Weight
        lut = []
        if inputIndex == minExposureOffsetIndex:
            lut.append(1)
        if inputIndex == maxExposureOffsetIndex:
            lut.append(2)
        if lut:
            print( "\tUsing LUT %s in weighting calculation" % lut )
        weightIntermediate = intermediate if writeIntermediate else 0
        ImageBufWeight(weight, inputBuffer, lut=lut)

        if writeIntermediate:
            intermediatePath = "%s_int%d.weight%s" % (inputPathComponents[0], intermediate, inputPathComponents[1])
            intermediate += 1
            ImageBufWrite(weight, intermediatePath)

        print( "\tMultiply by weight" )

        # Multiply color by weight
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
    ImageBufWrite(imageSum, outputPath)

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
    outputPath = None):
    print( "" )
    print( "Merge images into an HDR - begin" )
    print( "" )
    print( "Images : %s" % imageUris )

    # Set up other resources
    if baseExposureIndex is None:
        baseExposureIndex = -1
    else:
        print( "Base Exposure : %s" % baseExposureIndex )

    if not outputPath:
        outputPath = "%s%s" % (os.path.splitext( imageUris[0] )[0], ".exr")

    luts = []
    if responseLUTPath:
        responseLUT = loadResponse( responseLUTPath )
        luts = [responseLUTPath]

    # Merge the HDR images
    mkhdr(outputPath, imageUris, luts, baseExposureIndex, writeIntermediate)

    print( "" )
    print( "Merge images into an HDR - end" )
    print( "" )

def mergeHDRFolder(hdrDir, 
    responseLUTPath = None,
    writeIntermediate = False, 
    baseExposureIndex = None,
    outputPath = None):
    startingDir = os.getcwd()

    print( "mergeHDRFolder - folder : %s" % hdrDir )

    try:
        imageUris = sorted( os.listdir( hdrDir ) )
        imageUris = [x for x in imageUris if os.path.splitext(x)[-1].lower() in ['.jpg', '.cr2']]

        print( "mergeHDRFolder - images : %s" % imageUris )

        os.chdir( hdrDir )
        mergeHDRGroup( imageUris, 
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate,
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath )
    except Exception, e:
        print( "Exception in HDR merging" )
        print( repr(e) )

    os.chdir( startingDir )

#
# Get the options, load a set of images and merge them
#
def main():
    import optparse

    usage  = "%prog [options]\n"
    usage += "\n"

    p = optparse.OptionParser(description='Merge a set of LDR images into a single HDR result',
                                prog='mkhdr',
                                version='mkhdr',
                                usage=usage)

    p.add_option('--input', '-i', type='string', action='append')
    p.add_option('--inputFolder', default=None)
    p.add_option('--output', '-o', default=None)
    p.add_option('--responseLUT', '-r', default=None)
    p.add_option('--baseExposure', '-b', default=-1, type='int')
    p.add_option('--writeIntermediate', '-w', action="store_true")
    p.add_option('--verbose', '-v', action="store_true")

    options, arguments = p.parse_args()

    #
    # Get options
    # 
    inputPaths = options.input
    inputFolder = options.inputFolder
    outputPath = options.output
    responseLUTPath = options.responseLUT
    writeIntermediate = options.writeIntermediate == True
    verbose = options.verbose == True
    baseExposureIndex = options.baseExposure

    try:
        argsStart = sys.argv.index('--') + 1
        args = sys.argv[argsStart:]
    except:
        argsStart = len(sys.argv)+1
        args = []

    if verbose:
        print( "command line : \n%s\n" % " ".join(sys.argv) )

    # Process a folder 
    if inputFolder:
        mergeHDRFolder(inputFolder, 
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate, 
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath)

    # Process an explicit list of files
    else:
        mergeHDRGroup(inputPaths,
            responseLUTPath = responseLUTPath,
            writeIntermediate = writeIntermediate,
            baseExposureIndex = baseExposureIndex,
            outputPath = outputPath)

 # main

if __name__ == '__main__':
    main()
