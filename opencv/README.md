## OpenCV-based Python scripts ##

A set of scripts that wrap functionality from the OpenCV Python bindings.

These were tested with OpenCV version 3.3.1

Dependencies
-
This *Python* script depends on the following libraries:

- **OpenCV**: https://opencv.org/
- **OpenImageIO**: http://openimageio.org
    - Detailed build instructions can be found here: [OpenImageIO Build Instructions](https://sites.google.com/site/openimageio/checking-out-and-building-openimageio)

Building on Mac OSX
- 
Use the following commands to build this packages on Mac OSX

- Update the homebrew repository of install scripts to make sure that OpenImageIO is included.
    - brew tap homebrew/science
- Required Dependencies for OpenImageIO
    - brew install -vd libRaw
- Optional Dependencies for OpenImageIO
    - brew install -vd OpenCV
- OpenImageIO
    - brew install -vd openimageio --with-python
