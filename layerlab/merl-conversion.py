import os
import sys
sys.path.append('.')

import optparse

import layerlab as ll

def convertMerl(merl_sample_input, merl_sample_output, quadrature, quad_n=200, fourier_m=200):
    if not os.path.exists( merl_sample_output ):
        print( "Processing : %s" % merl_sample_input )
        if quadrature == "gaussLegendre":
            mu, w = ll.quad.gaussLegendre(quad_n)
        elif quadrature == "compositeSimpson":
            mu, w = ll.quad.compositeSimpson(quad_n)
        elif quadrature == "compositeSimpson38":
            mu, w = ll.quad.compositeSimpson38(quad_n)
        else:
            mu, w = ll.quad.gaussLobatto(quad_n)

        output = []
        for channel in range(3):
            # merl sampled data
            print("Creating merl layer")
            merl_layer = ll.Layer(mu, w, fourier_m)
            merl_layer.setMatusik(filename = merl_sample_input, channel = channel)

            output.append(merl_layer)

        # .. and write to disk
        print("Writing : %s" % merl_sample_output )

        storage = ll.BSDFStorage.fromLayerRGB(merl_sample_output, *output)
        storage.close()

def main():
    p = optparse.OptionParser(description='A script to convert MERL data to .bsdf files',
                              prog='merl-conversion',
                              version='merl conversion 0.1',
                              usage=('%prog [options]'))
    p.add_option('--inputFolder', '-i', default=None)
    p.add_option('--outputFolder', '-o', default=None)
    p.add_option('--file', '-f', default=None)
    p.add_option('--quadrature', '-q', default="gaussLobatto",
    	help="Valid options : gaussLegendre, compositeSimpson, compositeSimpson38, gaussLobatto, all. default=gaussLobatto")
    p.add_option('--quadrature_n', '-n', default=400,
    	help="default: 400")
    p.add_option('--fourier_m', '-m', default=200,
    	help="default: 200")

    options, arguments = p.parse_args()

    merl_brdfs_folder = options.inputFolder
    merl_brdf_file = options.file
    bsdf_output_folder = options.outputFolder
    quad_n = int(options.quadrature_n)
    fourier_m = int(options.fourier_m)
    quadrature = options.quadrature

    merl_samples = []
    if merl_brdfs_folder:
        print( "Merl Folder : %s" % merl_brdfs_folder )
        for dir_name, subdir_list, file_list in os.walk(merl_brdfs_folder):
            for fname in file_list:
                merl_samples.append(fname)
    else:
        print( "Merl File : %s" % merl_brdf_file )
        components = os.path.split( merl_brdf_file )
        merl_brdfs_folder = components[0]
        merl_samples.append(components[1])

    if not bsdf_output_folder:
        bsdf_output_folder = merl_brdfs_folder

    valid_quads = ["gaussLegendre", "compositeSimpson", "compositeSimpson38", "gaussLobatto"]
    if quadrature and quadrature in valid_quads:
        process_quads = [quadrature]
    elif quadrature == "all":
    	process_quads = valid_quads

    for merl_sample in merl_samples:
        print( "Processing Merl Sample : %s" % merl_sample )
        for quad in process_quads:
            print( "\tQuadrature : %s" % quad)
            try:
                merl_sample_input = os.path.join(merl_brdfs_folder, merl_sample)
                components = os.path.splitext( merl_sample )
                if len(process_quads) > 1:
                    merl_sample_output = "%s.%s.bsdf" % (components[0], quad)
                else:
                    merl_sample_output = "%s.bsdf" % components[0]
                merl_sample_output_abs = os.path.join(bsdf_output_folder, merl_sample_output)

                print( "\tInput  : %s\n\tOutput : %s" % (merl_sample_input, merl_sample_output_abs) )
                convertMerl( merl_sample_input, merl_sample_output_abs, quad, quad_n, fourier_m )
            except Exception, e:
                print( "\tException processing sample / quadrature combination" )
                print( repr(e) )

            print( "" )


if __name__ == '__main__':
    main()

