import sys
import os

import cv2
import numpy as np

import OpenImageIO as oiio
from OpenImageIO import ImageBuf

# Formats with exif data
generalExtensions = ["jpg", "tiff", "tif"]

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

def getShutterSpeed( imagePath,
	verbose = False ):
    imageBuffer = ImageBuf( imagePath )
    # Get attributes
    (channelType, width, height, channels, orientation, metadata, inputSpec) = ImageAttributes(imageBuffer)

    exposure = getExposureInformation(metadata)

    if verbose:
	    print( imagePath )
	    print( "\tChannel Type : %s" % (channelType) )
	    print( "\tWidth        : %s" % (width) )
	    print( "\tHeight       : %s" % (height) )
	    print( "\tChannels     : %s" % (channels) )
	    print( "\tOrientation  : %s" % (orientation) )
	    print( "\tExposure     : %s" % (exposure) )
	    print( "\tMetadata #   : %s" % (len(metadata)) )
	    print( "\tInv Shutter. : %s" % (exposure['ExposureTime']))

    return exposure['ExposureTime']

def recover_camera_response(inputFolder = '.', 
		exposureListFile = None,
    	outputResponseCurve = "camera_response.spi1d",
    	outputResponseFormat = "spi1d",
    	calibrationApproach = "berkeley",
    	mergeExposures = False,
    	mergedExposuresOutput = None,
    	verbose = False,
    	robertsonMaxIter = 30.0,
    	robertsonThreshold = 0.01,
    	berkeleyLambda = 20.0,
    	berkeleySamples = 1024,
    	berkeleySamplePlacementRandom = False):
	extensions = generalExtensions

	if exposureListFile:
		with open(exposureListFile, 'r') as f:
			exposuresList = f.readlines()

		exposuresList = [x.strip() for x in exposuresList if len(x) > 1]

		imageUris = [x.split(' ')[0] for x in exposuresList]		
		exposure_times = [1.0/float(x.split(' ')[1]) for x in exposuresList]

	else:		
		imageUris = sorted( os.listdir( inputFolder ) )
		imageUris = [x for x in imageUris if (os.path.splitext(x)[-1].lower()[1:] in extensions) and (x[0] != '.')]

		if verbose:
			print( imageUris )

		cwd = os.getcwd()
		os.chdir( inputFolder )

		exposure_times = [0]*len(imageUris)
		for i in range(len(imageUris)):
			exposure_times[i] = getShutterSpeed( imageUris[i], verbose=verbose )

	# List has to be sorted from longest shutter speed to shortest for opencv functions to work
	exposure_times, imageUris = (list(x) for x in zip(*sorted(zip(exposure_times, imageUris))))
	imageUris.reverse()

	exposure_times.reverse()
	exposure_times = np.array(exposure_times, dtype=np.float32)

	if verbose:
		for exposure in zip(exposure_times, imageUris):
			print( "Image : %s, Shutter speed : %2.6f" % (exposure[1], exposure[0]) )

	img_list = [cv2.imread(fn) for fn in imageUris ]

	if not exposureListFile:
		os.chdir( cwd )

	if calibrationApproach == "robertson":
		merge = cv2.createMergeRobertson()
		calibrate = cv2.createCalibrateRobertson()

		calibrate.setMaxIter(robertsonMaxIter)
		calibrate.setThreshold(robertsonThreshold)

		if verbose:
			print( calibrationApproach )
			print( "\tmax iter  : %d" % robertsonMaxIter )
			print( "\tthreshold : %f" % robertsonThreshold )
	else:
		merge = cv2.createMergeDebevec()
		calibrate = cv2.createCalibrateDebevec()

		calibrate.setLambda(berkeleyLambda)
		calibrate.setSamples(berkeleySamples)
		calibrate.setRandom(berkeleySamplePlacementRandom)

		if verbose:
			print( calibrationApproach )
			print( "\tlambda    : %3.2f" % berkeleyLambda )
			print( "\tsamples   : %d" % berkeleySamples )
			print( "\trandom    : %s" % berkeleySamplePlacementRandom )

	if verbose:
		print( "recovering camera response" )

	curve = calibrate.process(img_list, times=exposure_times)

	if verbose:
		print( "writing camera response - %s, %s" % (outputResponseFormat, outputResponseCurve) )

	if outputResponseFormat == "spi1d":
		with open(outputResponseCurve, "w") as f:
			f.write( "Version 1\n" )
			f.write( "From 0.000000 1.000000\n" )
			f.write( "Length 256\n" )
			f.write( "Components 3\n" )
			f.write( "{\n" )
			for i in range(len(curve)):
			    f.write( "%3.6f %3.6f %3.6f\n" % (curve[i][0][0]*0.18, curve[i][0][1]*0.18, curve[i][0][2]*0.18) )
			f.write( "}\n" )
	else:
		with open(outputResponseCurve, "w") as f:
			for i in range(len(curve)):
			    f.write( "%3.6f %3.6f %3.6f\n" % (curve[i][0][0], curve[i][0][1], curve[i][0][2]) )

	if mergedExposuresOutput:
		if verbose:
			print( "merging exposures" )

		hdr = merge.process(img_list, times=exposure_times.copy(), response=curve.copy())
		cv2.imwrite(mergedExposuresOutput, hdr)

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

    p.add_option('--inputFolder', '-i', default='.')
    p.add_option('--exposureListFile', '-e', default=None)
    p.add_option('--outputResponseCurve', '-o', default=None)
    p.add_option('--outputResponseFormat', '-f', default='spi1d', type='string',
        help="spi1d, raw")
    p.add_option('--calibrationApproach', '-c', default='berkeley', type='string',
        help="berkeley, robertson")
    p.add_option('--mergedExposuresOutput', '-m', default=None)
    p.add_option('--verbose', '-v', action="store_true")

    p.add_option('--robertsonMaxIter', default=30)
    p.add_option('--robertsonThreshold', default=0.01)

    p.add_option('--berkeleyLambda', default=20.0)
    p.add_option('--berkeleySamples', default=1024)
    p.add_option('--berkeleySamplePlacementRandom', default=False)

    options, arguments = p.parse_args()

    #
    # Get options
    # 
    inputFolder = options.inputFolder
    exposureListFile = options.exposureListFile
    outputResponseCurve = options.outputResponseCurve
    outputResponseFormat = options.outputResponseFormat
    calibrationApproach = options.calibrationApproach
    mergedExposuresOutput = options.mergedExposuresOutput
    verbose = options.verbose

    robertsonMaxIter = int(options.robertsonMaxIter)
    robertsonThreshold = float(options.robertsonThreshold)
    berkeleyLambda = float(options.berkeleyLambda)
    berkeleySamples = int(options.berkeleySamples)
    berkeleySamplePlacementRandom = options.berkeleySamplePlacementRandom == True

    try:
        argsStart = sys.argv.index('--') + 1
        args = sys.argv[argsStart:]
    except:
        argsStart = len(sys.argv)+1
        args = []

    if verbose:
        print( "command line : \n%s\n" % " ".join(sys.argv) )

    recover_camera_response(inputFolder = inputFolder, 
    	exposureListFile = exposureListFile,
    	outputResponseCurve = outputResponseCurve,
    	outputResponseFormat = outputResponseFormat,
    	calibrationApproach = calibrationApproach,
    	mergedExposuresOutput = mergedExposuresOutput,
    	verbose = verbose,
    	robertsonMaxIter = robertsonMaxIter,
    	robertsonThreshold = robertsonThreshold,
    	berkeleyLambda = berkeleyLambda,
    	berkeleySamples = berkeleySamples,
    	berkeleySamplePlacementRandom = berkeleySamplePlacementRandom)
# main

if __name__ == '__main__':
    main()

