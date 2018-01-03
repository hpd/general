import numpy as np

def createMatrixInverseNode(matrixNode):
    matrixValues = matrixNode['matrix'].getValue()
    a = np.array(matrixValues)
    a = a.reshape(3, 3)

    ainv = np.linalg.inv(a)
    
    inverseMatrixNode = nuke.createNode ("ColorMatrix")
    matrixInverseValues = ainv.reshape(1,9).tolist()[0]
    inverseMatrixNode['matrix'].setValue(matrixInverseValues)

selectedNode = nuke.selectedNode()
createMatrixInverseNode(selectedNode)

def combineMatrixNodes(matrixNode1, matrixNode2):
    matrixValues1 = matrixNode1['matrix'].getValue()
    a = np.array(matrixValues1)
    a = a.reshape(3, 3)
    
    matrixValues2 = matrixNode2['matrix'].getValue()
    b = np.array(matrixValues2)
    b = b.reshape(3, 3)
    
    c = np.dot(a, b)

    combinedMatrixNode = nuke.createNode ("ColorMatrix")
    combinedMatrixValues = c.reshape(1,9).tolist()[0]
    combinedMatrixNode['matrix'].setValue(combinedMatrixValues)

selectedNodes = nuke.selectedNodes()
combineMatrixNodes(selectedNodes[0], selectedNodes[1])

def writeSPIColorMatrix(matrixNode, spiFile):
    matrixValues = matrixNode['matrix'].getValue()
    
    with open(spiFile, "w") as spiFileHandle:
        spiFileHandle.write( "%3.6f %3.6f %3.6f 0.0\n" % (
            matrixValues[0], matrixValues[1], matrixValues[2]) )
        spiFileHandle.write( "%3.6f %3.6f %3.6f 0.0\n" % (
            matrixValues[3], matrixValues[4], matrixValues[5]) )
        spiFileHandle.write( "%3.6f %3.6f %3.6f 0.0\n" % (
            matrixValues[6], matrixValues[7], matrixValues[8]) )

gamutsFolder = "/Volumes/Titan/Simulator/Scenes/FremontVerticalSlice/Sequences/Shared/cameras/gamuts"
selectedNode = nuke.selectedNode()
spiFile = "%s/canonraw_to_srgb.spimtx" % gamutsFolder
writeSPIColorMatrix(matrixNode, spiFile):

selectedNode = nuke.selectedNode()
spiFile = "%s/Cam1_canonraw_to_srgb_20170427.spimtx" % gamutsFolder
writeSPIColorMatrix(selectedNode, spiFile)

def matrix3ToList(nukeMatrix3):
    matrixValues = []
    for i in range(9):
        matrixValues.append(nukeMatrix3[i])
    return matrixValues

def listToMatrix3(pyList):
    a = nuke.math.Matrix3()
    for i in range(9):
        a[i] = pyList[i]
    return a

def nodeToMatrix3(matrixNode):
    matrixValues = matrixNode['matrix'].getValue()
    nukeMatrix3 = nuke.math.Matrix3()
    for i in range(9):
        nukeMatrix3[i] = matrixValues[i]
    return nukeMatrix3

def matrix3ToNode(nukeMatrix3):
    matrixValues = []
    for i in range(9):
        matrixValues.append(nukeMatrix3[i])
    nukeMatrixNode = nuke.createNode ("ColorMatrix")
    nukeMatrixNode['matrix'].setValue(matrixValues)
    return nukeMatrixNode

def createMatrixInverseNodeNuke(matrixNode=None):
    if matrixNode == None:
        matrixNode = nuke.selectedNode()

    matrix = nodeToMatrix3(matrixNode)    
    matrixInverse = matrix.inverse()
    matrixInverseNode = matrix3ToNode(matrixInverse)

def combineMatrixNodesNuke(matrixNodes=None):
    if matrixNodes == None:
        selectedNodes = nuke.selectedNodes()
        matrixNodes = []
        matrixNode[0] = selectedNodes[0]
        matrixNode[1] = selectedNodes[1]

    a = nodeToMatrix3(matrixNodes[0])
    b = nodeToMatrix3(matrixNodes[1])

    invertMatrix = matrixNode[0]['invert'].getValue()
    if invertMatrix > 0.0:
        a = a.inverse()
    invertMatrix = matrixNode[1]['invert'].getValue()
    if invertMatrix > 0.0:
        b = b.inverse()
    
    c = a * b

    combinedMatrixNode = matrix3ToNode(c)

selectedNodes = nuke.selectedNodes()
combineMatrixNodesNuke(selectedNodes[0], selectedNodes[1])



