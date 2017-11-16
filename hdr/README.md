## mkhdr ##

A simple HDR exposure merging script based on OpenImageIO and Python

Dependencies
-
The *Python* configuration generation package depends on the following
libraries:

- **OpenImageIO**: http://openimageio.org
    - Detailed build instructions can be found here: [OpenImageIO Build Instructions](https://sites.google.com/site/openimageio/checking-out-and-building-openimageio)
- **OpenColorIO**: http://opencolorio.org
    - Detailed build instructions can be found here: [OpenColorIO Build Instructions](http://opencolorio.org/installation.html)


Building on Mac OSX
- 
Use the following commands to build these packages on Mac OSX

- OpenColorIO
    - brew install -vd opencolorio --with-python
- Update the homebrew repository of install scripts to make sure that OpenImageIO is included.
    - brew tap homebrew/science
- Required Dependencies for OpenImageIO
    - brew install -vd libRaw
- Optional Dependencies for OpenImageIO
    - brew install -vd OpenCV
- OpenImageIO
    - brew install -vd openimageio --with-python
- OpenColorIO, a second time. *ociolutimage* will build with *openimageio* installed.
    - brew uninstall -vd --ignore-dependencies opencolorio
    - brew install -vd opencolorio --with-python
