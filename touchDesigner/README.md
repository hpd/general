Touch Designer Utilities
=

Extensions to Touch Designer

3D LUT Support
-
To use the glsl in Touch Designer, follow these steps
- Create a GLSL TOP. 
	- Call this 'glsl_lookup_3d'
- Create a Text DAT. 
	- Call this 'glsl_source_lookup_3d'. This name is important.
- Create a MovieIn TOP. 
	- Load in your 3d LUT image.
	- The identity LUT image is in the 'scripts' folder. It is not the same as the Nuke CMSPattern LUT image.
- Connect the image that you want to apply the LUT to to your the GLSL TOP's first input
- Connect the LUT MovieIn TOP to the GLSL TOP's second input
- Open the GLSL TOP's parameters (as seen in the screenshots)
	- In the 'GLSL' parameter tab, type 'glsl_source_lookup_3d' into the Pixel Shader parameter
	- In the 'Common' parameter tab, switch 'Input Smoothness' to 'Nearest Pixel'

If you want to use a LUT with a different resolution or tile layout, swap out the parameter on the second to last line of the GLSL file. They are currently 32, 8, 4 for a 32x32x32 LUT that is laid out 8 tiles wide and 4 tiles high in the LUT image.


