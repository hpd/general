import cv2
import array
import numpy as np
import sys

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageSpec, ImageBufAlgo, ImageInput, ROI

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
        ImageBufAlgo.rotate90(imageBuf, imageBuf)
        ImageBufAlgo.reorient (imageBuf, imageBuf)

    elif orientation == 8:
        imageBuf.specmod().attribute ("Orientation", 1)
        ImageBufAlgo.rotate270(imageBuf, imageBuf)
        ImageBufAlgo.reorient (imageBuf, imageBuf)

    else:
        ImageBufAlgo.reorient (imageBuf, imageBuf)

def OpenCVImageBufferFromOIIOImageBuffer(oiioImageBuffer):
    oiioSpec = oiioImageBuffer.spec()
    (width, height, channels) = (oiioSpec.width, oiioSpec.height, oiioSpec.nchannels)
    oiioFormat = oiioSpec.format
    oiioChanneltype = oiioFormat.basetype

    #print( "OpenCVImageBufferFromOIIOImageBuffer", width, height, channels, oiioChanneltype )

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

    #print( "OIIOImageBufferFromOpenCVImageBuffer", width, height, channels, npChanneltype )

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

def applyOpticalFlow(img, flow):
    h, w = flow.shape[:2]
    base = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))
    pixel_map = np.array(base + -flow, dtype=np.float32)

    res = cv2.remap(img, pixel_map, None, cv2.INTER_LINEAR)
    return res

def findOpticalFlow(inputImage1, 
    inputImage2,
    outputWarpedImage,
    outputFlowImage,
    verbose,
    opticalFlowImplementation="simpleflow"):

    oiioImageBuffer1 = ImageBuf( inputImage1 )
    ImageBufReorient(oiioImageBuffer1, oiioImageBuffer1.orientation)

    oiioImageBuffer2 = ImageBuf( inputImage2 )
    ImageBufReorient(oiioImageBuffer2, oiioImageBuffer2.orientation)

    if verbose:
        print( "load and convert 1 - %s" % inputImage1 )
    openCVImageBuffer1 = OpenCVImageBufferFromOIIOImageBuffer(oiioImageBuffer1)
    if verbose:
        print( "load and convert 2 - %s" % inputImage2 )
    openCVImageBuffer2 = OpenCVImageBufferFromOIIOImageBuffer(oiioImageBuffer2)

    if verbose:
        print( "resolution : %s" % str(openCVImageBuffer1.shape) )
        print( "calculate optical flow 1 -> 2")

    if opticalFlowImplementation == "old_farneback":
        if verbose:
            print( "older farneback implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        previous_flow = None
        pyramid_scale = 0.5
        pyramid_levels = 5
        window_size = 50
        iterations_per_pyramid_level = 20
        pixel_neighborhood_size = 3
        neighborhood_match_smoothing_factor = 1.0
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

        if verbose:
            print( "calculate")
        opencvFlow = cv2.calcOpticalFlowFarneback(gray1, gray2, 
            previous_flow, pyramid_scale, pyramid_levels, window_size, iterations_per_pyramid_level, 
            pixel_neighborhood_size, neighborhood_match_smoothing_factor, flags)

    elif opticalFlowImplementation == "farneback":
        if verbose:
            print( "farneback implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        # Set of constants should be added
        implementation = cv2.optflow.createOptFlow_Farneback()
        if verbose:
            print( "calculate")
        opencvFlow = implementation.calc(gray1, gray2, None)

    elif opticalFlowImplementation == "dualtvl1":
        if verbose:
            print( "dualtvl1 implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        # Set of constants should be added
        implementation = cv2.createOptFlow_DualTVL1()
        if verbose:
            print( "calculate")
        opencvFlow = implementation.calc(gray1, gray2, None)

    elif opticalFlowImplementation == "sparsetodense":
        if verbose:
            print( "sparse to dense implementation" )
        # Current set of constants... Ranges and good values should be documented
        if verbose:
            print( "calculate")
        opencvFlow = cv2.optflow.calcOpticalFlowSparseToDense(openCVImageBuffer1, openCVImageBuffer2, None,
            8, 128, 0.05, True, 500.0, 1.5)

    elif opticalFlowImplementation == "deepflow":
        if verbose:
            print( "deep flow implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        # Set of constants should be added
        implementation = cv2.optflow.createOptFlow_DeepFlow()
        if verbose:
            print( "calculate")
        opencvFlow = implementation.calc(gray1, gray2, None)

    elif opticalFlowImplementation == "dis":
        if verbose:
            print( "dis implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        # Set of constants should be added
        implementation = cv2.optflow.createOptFlow_DIS()
        if verbose:
            print( "calculate")
        opencvFlow = implementation.calc(gray1, gray2, None)

    elif opticalFlowImplementation == "pcaflow":
        if verbose:
            print( "pca flow implementation" )

        if verbose:
            print( "to grey 1")
        gray1 = cv2.cvtColor(openCVImageBuffer1, cv2.COLOR_BGR2GRAY)

        if verbose:
            print( "to grey 2")
        gray2 = cv2.cvtColor(openCVImageBuffer2, cv2.COLOR_BGR2GRAY)

        # Set of constants should be added
        implementation = cv2.optflow.createOptFlow_PCAFlow()
        if verbose:
            print( "calculate")
        opencvFlow = implementation.calc(gray1, gray2, None)

    elif opticalFlowImplementation == "simpleflow":
        if verbose:
            print( "simple flow implementation" )
        # Current set of constants... Ranges and good values should be documented
        opencvFlow = cv2.optflow.calcOpticalFlowSF(openCVImageBuffer1, openCVImageBuffer2, 
            3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10)

    else:
        print( "Unknown optical flow implementation : %s" % opticalFlowImplementation )
        opencvFlow = None

    if outputWarpedImage and (opencvFlow is not None):
        if verbose:
            print( "warping 1 -> 2")
        opencvWarped = applyOpticalFlow(openCVImageBuffer1, opencvFlow)

        if verbose:
            print( "converting and writing warped image - %s" % outputWarpedImage )
        oiioWarped = OIIOImageBufferFromOpenCVImageBuffer( opencvWarped )
        oiioWarped.write( outputWarpedImage )
    else:
        opencvWarped = None

    if outputFlowImage and (opencvFlow is not None):
        if verbose:
            print( "converting and writing flow image - %s" % outputFlowImage )

        oiioFlowBuffer = OIIOImageBufferFromOpenCVImageBuffer( opencvFlow )
        oiioFlowBuffer.write( outputFlowImage )

    return (opencvWarped, opencvFlow)

#
# Get the options, load a set of images and merge them
#
def main():
    import optparse

    usage  = "%prog [options]\n"
    usage += "\n"

    p = optparse.OptionParser(description='Recover the camera response curve from a set of exposures',
                                prog='recoverCameraResponse',
                                version='1.0',
                                usage=usage)

    p.add_option('--inputImage1', default=None)
    p.add_option('--inputImage2', default=None)
    p.add_option('--outputWarpedImage', default=None)
    p.add_option('--outputFlowImage', default=None)
    p.add_option('--opticalFlowImplementation', default="deepflow")

    p.add_option('--verbose', '-v', action="store_true")

    options, arguments = p.parse_args()

    #
    # Get options
    # 
    inputImage1 = options.inputImage1
    inputImage2 = options.inputImage2
    outputWarpedImage = options.outputWarpedImage
    outputFlowImage = options.outputFlowImage
    verbose = options.verbose
    opticalFlowImplementation = options.opticalFlowImplementation

    try:
        argsStart = sys.argv.index('--') + 1
        args = sys.argv[argsStart:]
    except:
        argsStart = len(sys.argv)+1
        args = []

    if verbose:
        print( "command line : \n%s\n" % " ".join(sys.argv) )

    if inputImage1 and inputImage2:
        findOpticalFlow(inputImage1, inputImage2, outputWarpedImage, outputFlowImage, verbose,
            opticalFlowImplementation=opticalFlowImplementation)
    else:
        print( "\nTwo input images must be supplied.\n" )
        usage()
# main

if __name__ == '__main__':
    main()
