'''
A script to create Maya lights and cameras from Otoy light stage data files

Usage:

import os
import sys

sys.path.append( "/path/to/script" )
import mayaLightStageImport as mlsi

lightStageData = "/path/to/lightStage/data"
cameraDir = os.path.join( lightStageData, "CH2_cameras" )
lightingDir = os.path.join( lightStageData, "LS_lighting_info" )

# Create lights, key frames to match light stage key frames
lightStageGroup = mlsi.createLightStageLights( lightingDir )

# Create cameras
cameraGroup = mlsi.createLightStageCameras(cameraDir)
'''

import math
import os
import sys

def parseCamera(cameraFile):
	camera = {}

	with open(cameraFile, 'r') as fileHandle:
		lines = fileHandle.readlines()

		for line in lines:
			tokens = line.strip().split('=')
			camera[tokens[0].strip().lower()] = map(float, tokens[1].split())

	return camera

def readCameras(camerasDir):
	cameraData = {}

	for fileName in os.listdir(camerasDir):
		if fileName.startswith("cam"):
			print(fileName)

			cameraName = fileName.split('.')[0]
			camera = parseCamera(os.path.join(camerasDir, fileName))
			cameraData[cameraName] = camera

	return cameraData

def parseLightDirections(directionsFile):
	print(directionsFile)

	directions = {}

	with open(directionsFile, 'r') as fileHandle:
		lines = fileHandle.readlines()

		for line in lines:
			tokens = line.strip().split()
			directions[int(tokens[0].strip())] = map(float, tokens[1:])

	return directions

def parseLightPolarization(polarizationFile):
	print(polarizationFile)

	polarization = {}

	with open(polarizationFile, 'r') as fileHandle:
		lines = fileHandle.readlines()

		for line in lines:
			tokens = line.strip().split()
			polarization[int(tokens[0].strip())] = int(tokens[1].strip())

	return polarization

def parseLightConfigurations(configurationsFile):
	print(configurationsFile)

	configurations = {}

	with open(configurationsFile, 'r') as fileHandle:
		lines = fileHandle.readlines()

		i = 1
		for line in lines:
			tokens = line.strip().split()
			configurations[i] = map(int, tokens)
			i += 1

	return configurations


def readLights(lightingDir):
	lightData = {}

	for fileName in os.listdir(lightingDir):
		print(fileName)

		if fileName.startswith("directions"):
			directions = parseLightDirections(os.path.join(lightingDir, fileName))

			lightData['directions'] = directions

		elif fileName.startswith("is_a_vertex"):
			polarization = parseLightPolarization(os.path.join(lightingDir, fileName))

			lightData['polarization'] = polarization

		elif fileName.startswith("reference_lighting"):
			configurations = parseLightConfigurations(os.path.join(lightingDir, fileName))

			lightData['configurations'] = configurations

	return lightData

import maya.cmds as cmds

def inchesToCentimeters(inches):
    return inches*2.54

def halfInchesToCentimeters(inches):
    return inches*1.27

def createLightStageLight(name, direction, diameterIn, distanceToFrontIn, useGroup=True):
    diameterCm = inchesToCentimeters(diameterIn)
    distanceToFrontCm = inchesToCentimeters(distanceToFrontIn)

    sphereLight = cmds.polySphere(name=name, r=0.5, sx=20, sy=20, ax=[0, 1, 0], cuv=2, ch=1)[0]
    lightScale = diameterCm
    cmds.setAttr("%s.%s" % (sphereLight, "scaleX"), lightScale)
    cmds.setAttr("%s.%s" % (sphereLight, "scaleY"), lightScale)
    cmds.setAttr("%s.%s" % (sphereLight, "scaleZ"), lightScale)

    if useGroup:
        cmds.setAttr("%s.%s" % (sphereLight, "scaleZ"), 0.0)

        lightGroup = cmds.group(sphereLight, name=("%sRotation" % sphereLight))

        lightTranslate = distanceToFrontCm
        cmds.setAttr("%s.%s" % (sphereLight, "translateZ"), lightTranslate)

        rx = -math.asin( direction[1] )*180.0/3.14159
        ry = math.atan2( direction[0], direction[2] )*180.0/3.14159

        cmds.setAttr("%s.%s" % (lightGroup, "rotateX"), rx)
        cmds.setAttr("%s.%s" % (lightGroup, "rotateY"), ry)

        return lightGroup
    else:
	    lightTranslate = map( lambda x: x*(distanceToFrontCm + diameterCm/2.), direction)
	    cmds.setAttr("%s.%s" % (sphereLight, "translateX"), lightTranslate[0])
	    cmds.setAttr("%s.%s" % (sphereLight, "translateY"), lightTranslate[1])
	    cmds.setAttr("%s.%s" % (sphereLight, "translateZ"), lightTranslate[2])

	    return sphereLight

def setLightConfigurationVisibility(lightConfigurations, lightNumber, name):
    for frame, configuration in lightConfigurations.iteritems():
        #print( frame )
        visible = ( lightNumber in configuration )
        #print( visible )
        cmds.currentTime( frame )
        cmds.setAttr("%s.%s" % (name, "v"), visible)
        cmds.setKeyframe("%s.%s" % (name, "v"))

def createLightStageLights(lightingDir):
	diameterIn = 4
	distanceToFrontIn = 55

	lightData = readLights(lightingDir)
	for dict, value in lightData.iteritems():
	    print( dict )
	    print( value )

	lightDirections = lightData['directions']
	lightPolarizations = lightData['polarization']
	lightConfigurations = lightData['configurations']

	lightsPolarized = []
	lightsUnpolarized = []
	for lightNumber, lightDirection in lightDirections.iteritems():
	    print( lightNumber, lightDirection )
	    if lightPolarizations[lightNumber] == 0:
	        name = "lightStageLightUnPolarized" + str(lightNumber)
	        lightsPolarized.append( createLightStageLight(name, lightDirection, diameterIn, distanceToFrontIn) )
	        setLightConfigurationVisibility(lightConfigurations, lightNumber, name)
	    else:    
	        name = "lightStageLightPolarized" + str(lightNumber)
	        lightsUnpolarized.append( createLightStageLight(name, lightDirection, diameterIn, distanceToFrontIn) )
	        setLightConfigurationVisibility(lightConfigurations, lightNumber, name)

	polarizedLightsGroup = cmds.group( lightsPolarized, name="polarizedLights" )
	unpolarizedLightsGroup = cmds.group( lightsUnpolarized, name="polarizedLights" )
	lightStageGroup = cmds.group( [polarizedLightsGroup, unpolarizedLightsGroup], name="lightStageLights" )

	return lightStageGroup

def createLightStageCamera(name, rx, ry, rz, translate, focalLengthPixels, ppX, ppY):
	c = cmds.camera(name=name)[0]

	# Rotation
	rotationMatrix = [0.0]*16
	#for i in range(0,3): rotationMatrix[i] = rx[i]
	#for i in range(0,3): rotationMatrix[i+4] = ry[i]
	#for i in range(0,3): rotationMatrix[i+8] = rz[i]

	# Transpose the rotation matrix values
	for i in range(0,3): rotationMatrix[i*4  ] = rx[i]
	for i in range(0,3): rotationMatrix[i*4+1] = ry[i]
	for i in range(0,3): rotationMatrix[i*4+2] = rz[i]
	rotationMatrix[15] = 1.0

	cmds.xform( c, a=True, matrix=rotationMatrix )

	# Translation
	t = map( halfInchesToCentimeters, translate )
	cmds.xform( c, a=True, translation=t )

	# Scale
	#cmds.xform( c, a=True, scale=[10.0, 10.0, 10.0] )
	
	# Film Back
	cmds.setAttr( "%s.verticalFilmAperture" % c, 1.417)
	cmds.setAttr( "%s.horizontalFilmAperture" % c, 0.9449)
	cmds.setAttr( "%s.cap" % c, l=True )
	cmds.setAttr( "%s.filmFit" % c, 2)

	# Focal Length
	sensorWidthMM = 24.0

	imageWidthPixels = 3456
	imageHeightPixels = 5184

	focalLengthMM = focalLengthPixels * sensorWidthMM / imageWidthPixels
	cmds.setAttr( "%s.focalLength" % c, focalLengthMM)

	# Film Back Offset
	centerX = imageWidthPixels/2.
	offsetX = (centerX - ppX)/imageWidthPixels
	cmds.setAttr( "%s.horizontalFilmOffset" % c, offsetX)

	centerY = imageHeightPixels/2.
	offsetY = (centerY - ppY)/imageHeightPixels
	cmds.setAttr( "%s.verticalFilmOffset" % c, offsetY)

	return c

def createLightStageCameras(cameraDir):
    cameraData = readCameras(cameraDir)

    mayaCameras = []
    for cam, data in cameraData.iteritems():
        mayaCamera = createLightStageCamera(cam, data['rx'], data['ry'], data['rz'], 
        	data['t'], data['fc'][0], data['pp'][0], data['pp'][1]) 
        mayaCameras.append( mayaCamera )

    cameraGroup = cmds.group( mayaCameras, name="lightStageCameras" )
    return cameraGroup





