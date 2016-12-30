#!/usr/bin/env python

# file:///net/dl1/mighell/aas228/dlkm_aas228_code.py

## IPython magic: forced reset
#%reset -f

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from astropy.table import Table
from astropy import convolution
import os
import os.path
import sys
import urllib

def get_file(
url=None,
filename=None,
check=True
):
    assert url is not None, "get_file: url must be defined!"
    assert filename is not None, "get_file: filename must be defined!"
    skip = False
    if not os.path.isfile(filename):
        urllib.urlretrieve(url,filename)
    else:
        print 'File exists %s  :-)' % (filename)
        skip = True
    if check and not skip:
        if not os.path.isfile(filename):
            print '\n8=X  ***** ERROR *****  File does not exist: %s\n' % (filename)
            sys.exit()
        else:
            print 'Retrieved file %s  :-)' % (filename)
        pass
    pass
    return 0
pass

def sia_vot_get_gr_files( 
sia_votable_filename='get.vot',
single=True
):
    """
    Reads a NOAO DataLab SIA service VOTable and retrieves the g & r observations
    """

    # Read the SIA VOTable
    filename = sia_votable_filename
    if not os.path.isfile(filename):
        print '\n***** ERROR ***** File does not exist: %s\n' % (filename)
        sys.exit()
    else:
        print 'Read file %s' % (filename)
    pass
    t = Table.read(sia_votable_filename)

    # Parse the table to get only images and dqmasks for the g and r observations
    count = 0
    je = len(t)
    for j in xrange(0,je):
        url = t[j]['access_url']
        b = url.find("=") + 1
        foo = url[b:]
        e = foo.find("&")
        bar = foo[:e]
        if "ooi" in bar:
            if ("_g_" in bar) or ("_r_" in bar):
                e = bar.index('.fits.fz')
                if single:
                    url = url[:url.index('&POS')]
                    ofile = bar[:e] + '_x.fits.fz'
                else:
                    url = url[:url.index('&extn')]
                    ofile = bar[:e] + '_.fits.fz'
                pass
                get_file(url, ofile)
                url2 = url.replace("ooi", "ood")
                ofile2 = ofile.replace("ooi", "ood")
                get_file(url2, ofile2)
                count += 1
            pass
        pass
    pass

    return count # how many observations were found
pass

#standalone: if __name__ == '__main__':
def get_eri2(
sia_vot_filename=None,
single=True
):
    """
    Use the prototype NOAO DataLab Simple Image Access (SIA) service to 
    generate a VOTable and retrieve the fpacked g & r filter observations 
    of the Eridanus II dwarf galaxy.

    Usage:

    get_eri2( sia_votable_filename='sia.vot', single=True )

    Example:

    get_eri2( 'sia.vot' )
    """

    # Submit a query to the prototype NOAO SIA service (result should be a VOTable)
    assert sia_vot_filename is not None, "get_eri2: sia_vot_filename must be defined!"
    # Ignore messeges sent to stderr # <-- Does not work in jupyter IPython notebooks :-(
    f = open(os.devnull, 'w')
    sys.stderr = f
    filename = sia_vot_filename #standalone: filename = sys.argv[1]

    # Where is the dwarf galaxy?
    """
    EridanusII dwarf galaxy: 
    (56.0878, -43.5332) [degrees]
    (03h44m21.07s, -43d31'59.5")
    """
    print 'Find observations of the Eridanus dwarf galaxy using the prototype NOAO SIA service:'
    get_file("http://zeus1.sdm.noao.edu/siapv1?POS=56.0878,-43.5332&SIZE=0.2", filename)
    print 'Search complete :-)'
    
    print 'Retrieve the g & r observations:'
    count = sia_vot_get_gr_files(sia_votable_filename=filename,single=single)
    print count, 'observations retrieved :-)'

    # Sign off
    print "\nThat's all Folks!\n"
pass

def get_eri2_big_fits(
base_url='http://dldb1.sdm.noao.edu/mighell/aas228/nb/eri2/'
):
    list = [
    'c4d_131109_072742_ood_g_d1_.fits',
    'c4d_131109_072742_ooi_g_d1_.fits',
    'c4d_131203_065201_ood_g_d1_.fits',
    'c4d_131203_065201_ooi_g_d1_.fits',
    'c4d_131203_065400_ood_r_d1_.fits',
    'c4d_131203_065400_ooi_r_d1_.fits',
    'c4d_131228_034807_ood_g_d1_.fits',
    'c4d_131228_034807_ooi_g_d1_.fits',
    'c4d_131231_052535_ood_r_d1_.fits',
    'c4d_131231_052535_ooi_r_d1_.fits',
    'c4d_140103_045004_ood_g_d1_.fits',
    'c4d_140103_045004_ooi_g_d1_.fits',
    'c4d_140103_045419_ood_r_d1_.fits',
    'c4d_140103_045419_ooi_r_d1_.fits']
    for filename in list: 
        get_file( base_url+filename, filename)
pass

def get_eri2_small_fits(
base_url='http://dldb1.sdm.noao.edu/mighell/aas228/nb/eri2/'
):
    list = [
    'c4d_131109_072742_ood_g_d1_x.fits',
    'c4d_131109_072742_ooi_g_d1_x.fits',
    'c4d_131203_065201_ood_g_d1_x.fits',
    'c4d_131203_065201_ooi_g_d1_x.fits',
    'c4d_131203_065400_ood_r_d1_x.fits',
    'c4d_131203_065400_ooi_r_d1_x.fits',
    'c4d_131228_034807_ood_g_d1_x.fits',
    'c4d_131228_034807_ooi_g_d1_x.fits',
    'c4d_131231_052535_ood_r_d1_x.fits',
    'c4d_131231_052535_ooi_r_d1_x.fits',
    'c4d_140103_045004_ood_g_d1_x.fits',
    'c4d_140103_045004_ooi_g_d1_x.fits',
    'c4d_140103_045419_ood_r_d1_x.fits',
    'c4d_140103_045419_ooi_r_d1_x.fits']
    for filename in list: 
        get_file( base_url+filename, filename)
pass

def plot_cmd( 
color=None,
mag=None,
fignum=None, 
extent=None, 
cmap=cm.Greys, 
title='', 
filename='',
xsize=8.0,
ysize=8.0,
xlim=None, 
ylim=None,
xtitle=None,
ytitle=None,
size=10,
verbose=False
):
    # A function to plot a color-magnitude diagram
    assert (color.size == mag.size), 'The size of color and mag must be the same!'
    fig = plt.figure(fignum)
    if verbose:
        print 'Figure ', fignum
        # color='gray'
    plt.scatter( color, mag, color='deepskyblue', s=size, alpha=0.05, edgecolor='blue')
    #plt.plot( color, mag, 'bo', markersize=3 )
    if xtitle is None:
        xtitle=r'Color [mag]'
    plt.gca().set_xlabel(xtitle, fontsize=16)
    if ytitle is None:
        ytitle=r'Magnitude [mag]'
    plt.gca().set_ylabel(ytitle, fontsize=16)
    if xlim is not None:
        plt.gca().set_xlim(xlim) # e.g. xlim=(-5,5)
    if ylim is not None:
        plt.gca().set_ylim(ylim) # e.g. ylim=(0,30)
    plt.gca().invert_yaxis()
    plt.grid(True,color='black',linewidth=1)
    plt.gcf().set_size_inches(xsize,ysize)
    if len(title) != 0:
        fig.suptitle(title, fontsize=18)
    if len(filename) != 0:
        plt.savefig(filename) # filename='foo.png' creates a PNG file. 
        if verbose:
            print 'Saved as ', filename
    plt.show()
    return
pass

def plot_hist2d( 
hist2d, 
xlim=None, 
ylim=None, 
fignum=None, 
extent=None, 
cmap=cm.Greys, 
title='', 
filename='',
verbose=False
):
    # A function to display 2-d histograms
    fig = plt.figure(fignum)
    if verbose:
        print 'Figure ', fignum
    plt.imshow(hist2d,extent=extent,interpolation='nearest', cmap=cmap)
    if ylim is not None:
        plt.gca().set_ylim(xlim) # e.g. ylim=(-5,5)
    if xlim is not None:
        plt.gca().set_xlim(ylim) # e.g. xlim=(225,235)
    plt.gca().invert_xaxis()
    plt.grid(True,color='black',linewidth=1)
    plt.gca().set_xlabel(r'Right Ascension: $\alpha$ [degrees]', fontsize=16)
    plt.gca().set_ylabel(r'Declination: $\delta$ [degrees]', fontsize=16)
    if len(title) != 0:
        fig.suptitle(title, fontsize=18)
    if len(filename) != 0:
        plt.savefig(filename) # filename='foo.png' creates a PNG file. 
        if verbose:
            print 'Saved as ', filename
    plt.colorbar()
    if verbose:
        print 'maximum value is', hist2d.max()
        print 'minimum value is', hist2d.min()
    plt.show()
    return
pass

def dlkm_aas228_dwarf_filter(
ra_degree,
dec_degree,
fwhm_small_arcmin=3,
fwhm_big_arcmin=30,
show_plots=False,
return_results=False,
floor_sigma=2.0,
ceiling_sigma=7.0
):
    assert (ra_degree.size == dec_degree.size), 'ra_degree.size and dec_degree.size must be the same!'
    x = ra_degree
    y = dec_degree
    # Information about declination (y) [degrees]
    ymin = y.min() 
    ymax = y.max() 
    ydiff = ymax - ymin 
    ymean = (ymin+ymax)/2.0
    ydiff_arcmin = ydiff * 60.0 # convert from degrees to arcmin
    # Information about right ascension (x) [degrees (time)]:
    xmin = x.min() 
    xmax = x.max() 
    xdiff = xmax - xmin 
    xmean = (xmin+xmax)/2.0
    # Deal with cosine(declination) compression
    xdiff_angular = xdiff * np.cos(ymean*(np.pi/180.0)) 
    xdiff_angular_arcmin = xdiff_angular * 60.0 # convert from degress to arcmin
    # Number of one-arcmin pixels in the X and Y directions:
    nx=np.rint(xdiff_angular_arcmin).astype('int')
    ny=np.rint(ydiff_arcmin).astype('int')
    # Create a two-dimensional histogram
    Counts, xedges, yedges  = np.histogram2d (x, y, (nx,ny) )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    c = np.rot90(Counts).copy() # hack around Pythonic weirdness
    if show_plots:
        plot_hist2d(c, extent=extent, fignum=1, title=title+' (raw counts)', 
            filename='dlkm_aas228_dwarf_filter_fig1.eps')
    # Make a small Gaussian kernel with a standard deviation of stddev_small_px pixels:
    stddev_small_px = fwhm_small_arcmin/2.35 # assuming arcmin^2 pixels
    kernel_small = convolution.Gaussian2DKernel(stddev=stddev_small_px,factor=1)
    # Make a big Gaussian kernel with a standard deviation of stddev_big_px pixels:
    stddev_big_px = fwhm_big_arcmin/2.35 # assuming arcmin^2 pixels
    kernel_big = convolution.Gaussian2DKernel(stddev=stddev_big_px,factor=1)
    # Differential Gaussian convolution kernel
    c_big = convolution.convolve(c,kernel_big)
    c_small = convolution.convolve(c,kernel_small)
    c_delta = c_small - c_big
    d = c_delta.copy()
    if show_plots:
        plot_hist2d(d, extent=extent, fignum=2, title=title+' (differential Gaussian convolution kernel)', 
            filename='dlkm_aas228_dwarf_filter_fig2.eps' )
    if show_plots: # 1-d histogram of filtered pixels
        dv = np.ndarray.flatten(d)
        plt.figure(3)
        plt.hist(dv,100,color='lightgrey')
        plt.xlabel('cell value')
        plt.ylabel('counts per cell')
        plt.savefig('dlkm_aas228_dwarf_filter_fig3.eps')
        plt.show()
    # Compute statistics 
    mean = np.mean(d,dtype='float64')
    sigma = np.std(d,dtype='float64')
    median = np.median(d)
    floor = mean + (floor_sigma*sigma)
    ceiling = mean + (ceiling_sigma*sigma)
    if show_plots: # statistics
        print d.size, '=d.size (number of objects)'
        print np.amin(d), '=minimum'
        print np.amax(d), '=maximum'
        print median, '=median'
        print mean, '=mean'
        print sigma, '=sigma'
        print floor, '=floor'
        print ceiling, '=ceiling'
    # Clip the values in the filtered 2-d histogram:
    e = d.copy()
    e[e>ceiling] = ceiling
    e[e<floor] = floor
    if show_plots:
        plot_hist2d(e, extent=extent, fignum=4, title=title+' (clipped differential Gaussian convolution kernel)', 
            filename='dlkm_aas228_dwarf_filter_fig4.eps' )
    hist2d_counts = c
    hist2d_filter = d
    hist2d_clipped = e
    if return_results:
        return hist2d_counts, extent, hist2d_filter, hist2d_clipped
    else:
        return
pass

def skyplot_scatter( 
ra, 
dec, 
title='', 
filename='',
size=10
):
    fig, ax = plt.subplots()
    #ax.scatter( ra, dec, color='gray', s=2, alpha=0.05)
    ax.scatter( ra, dec, color='deepskyblue', edgecolor='blue', s=size, alpha=0.05)
    ax.set_xlabel(r'Right Ascension: $\alpha$', fontsize=16)
    ax.set_ylabel(r'Declination: $\delta$', fontsize=16)
    ax.set_ylim([dec.min(),dec.max()])
    ax.set_xlim([ra.min(),ra.max()])
    ax.invert_xaxis()
    if len(title) != 0:
        fig.suptitle(title, fontsize=18)
    fig.set_size_inches(9.0,9.0)
    plt.grid(color='cyan',linewidth=1)
    if len(filename) != 0:
        plt.savefig(filename) # filename='foo.png' creates a PNG file. 
        print 'Saved as ', filename
    plt.show()
pass

#EOF
