from __future__ import absolute_import, unicode_literals, division, print_function

import numpy as np

import six

"""
Scripts for reducing Palomar Doublespec data
"""


def iraf_sec_to_slice(s, offsets=None):
    """
    Converts an IRAF-style section specifier to a python slice object

    offsets are ((st0, end0), (st1, end1), ...)
    """
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]

    sls = []
    for i, si in enumerate(s.split(',')):
        a, b = si.split(':')
        if offsets is not None:
            a = int(a) - 1 + offsets[i][0]
            b = int(b) + offsets[i][1]
            sls.append(slice(a, b))
        else:
            sls.append(slice(int(a) - 1, int(b)))

    return tuple(sls)

def _linfit(model, x, y):
    # used inside subtract_sky to fit a linear model to the sky, which is
    # much faster

    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    xb = np.mean(x)
    yb = np.mean(y)
    slope = (np.mean(x * y) - xb * yb) / (np.mean(x**2) - xb**2)
    intercept = yb - slope * xb
    return model.__class__(slope=slope, intercept=intercept)


class DoubleSpecImage(object):
    """
    A class for holding an image from the Doublespec spectrograph at the Hale

    Parameters
    ----------
    data : str or array
        Either a string pointing to a filename (in which case other Parameters
        should be None), or an array with image data
    unc : array or None
        Uncertainties (i.e., standard deviation) by pixel or None if `data` is
        a string (they will be inferred from read noise and poisson stats)
    header : astropy.io.fits.Header or None
        header from the source data file
    side : str or None
        'blue', 'red', or None to guess based on the filename of `data`
    """
    def __init__(self, data, unc=None, header=None, side=None):
        from warnings import warn
        from astropy.io import fits

        if isinstance(data, six.string_types):
            with fits.open(data) as f:
                if side is None:
                    if data.split('/')[-1].startswith('red'):
                        self.side = 'red'
                    elif data.split('/')[-1].startswith('blue'):
                        self.side = 'blue'
                    else:
                        raise ValueError("couldn't figure out red vs. blue side from file name")
                self.header = f[0].header
                self.data = f[0].data.astype(float)  # weird stuff happens if its not a float

                #start with error as just readout noise in ADU
                unc = self.header['RON'] / self.header['GAIN']
                self.unc = unc * np.ones_like(self.data)

            if self.header['ROISEC'] not in ('[910:1269,590:3424]',
                                             '[1:4114,785:1224]'):
                warn("ROISEC is not one of the values this pipeline assumes."
                     "It may not work right where specific values are used "
                     "(e.g., creating master flats")
        else:
            self.data = data
            self.unc = unc
            self.header = header
            self.side = side

        self.processingdone = tuple()  # this is replaced as processing happens

        #check they're valid
        for var in ('data', 'unc', 'header'):
            if getattr(self, var) is None:
                raise ValueError(var + ' cannot be None')
        if self.side not in ('red', 'blue'):
            raise ValueError('Need to specify red or blue side!')

    def hist(self, **kwargs):
        """
        Matplotlib histogram of the data
        """
        from matplotlib import pyplot as plt

        kwargs.setdefault('histtype', 'step')
        kwargs.setdefault('bins', 100)
        kwargs.setdefault('log', True)

        res = plt.hist(self.data.ravel(), **kwargs)
        #sometimes this doesn't auto-limit like it ought to
        if 'range' in kwargs:
            plt.xlim(*kwargs['range'])
        else:
            plt.xlim(self.data.min(), self.data.max())
        return res

    def count_plot(self, **kwargs):
        """
        Matplotlib plot of the rounded integer counts of the data
        """
        from matplotlib import pyplot as plt

        v = np.round(self.data.ravel()).astype(int)
        cnts = np.concatenate((np.bincount(v), [0]))
        x = np.arange(len(cnts))

        kwargs.setdefault('linestyle', 'steps-mid')
        kwargs.setdefault('log', True)

        logify = kwargs.pop('log', False)

        if logify:
            cnts = cnts.astype(float)
            cnts[cnts == 0] = .1
            res = plt.semilogy(x, cnts, **kwargs)
        else:
            res = plt.plot(x, cnts, **kwargs)
        plt.xlim(0, x[-1])

        return res

    def show_image(self, **kwargs):
        """
        Matplotlib imshow of this image

        transpose image for blue side unless transpose=False is given

        if log=True is given,the whole image will be linearly rescaled to 1-100
        before log-ifying, possibly be vmin/vmax
        """
        from matplotlib import pyplot as plt

        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('transpose', self.side=='blue')

        if kwargs.pop('log', False):
            safedata = self.data[np.isfinite(self.data)]
            #use vmin and vmax to rescale  1-100
            lmin = kwargs.pop('logmin', None)
            lmax = kwargs.pop('logmax', None)
            mx = lmax if lmax else np.max(safedata)
            mi = lmin if lmin else np.min(safedata)

            rescaleddata = ((self.data - mi) / (mx - mi))
            rescaleddata[rescaleddata>1] = 1
            rescaleddata[rescaleddata<0] = 0

            d = np.log10(rescaleddata*99.+1.)
        else:
            d = self.data

        if kwargs.pop('transpose'):
            d = d.T

        res = plt.imshow(d, **kwargs)
        return res

    def show_in_ds9(self, ds9=None, frame=None, unc=True):
        """
        Shows this image in ds9 using pyds9.

        Parameters
        ----------
        ds9 : pyds9.ds9 object or None
            The ds9 object to use or None to create a new one
        frame : int or None
            The frame to use to show the image
        unc : bool
            If true, will show the uncertainty on the frame after this one
        """
        if ds9 is None:
            import pyds9
            ds9 = pyds9.DS9()

        if frame is not None:
            ds9.set('frame ' + str(frame))
        ds9.set_np2arr(np.array(self.data, copy=True, order='C').T)

        if unc:
            ds9.set('frame {0}'.format(int(ds9.get('frame')) + 1))
            ds9.set_np2arr(np.array(self.unc, copy=True, order='C').T)

    @property
    def objname(self):
        return self.header['OBJECT']


    def overscan_sub_and_trim(self, biassecoffsets=((5, 0), (0, 0)),
                              splinesmoothing=None):
        """
        Do overscan subtraction and trimming.

        Parameters
        ----------
        biassecoffsets : 2-tuple of 2-tuples
            A list of index offsets to padd the overscal region that is actually
            fit - in the form ((xofflow, xoffhigh), (yofflow, yoffhigh))
        splinesmoothing : None or float
            The factor used by UnivariateSpline to decide how smooth to make the
            spline.  0 means interpolating spline, None means scipy default.

        Returns
        -------
        newimg : DoubleSpecImage
            A new image with the overscan subtracted and trimmed and the
            uncertainty updated to match
        """
        from scipy.interpolate import UnivariateSpline

        #need to swap order because FITS x/y is transpose of numpy
        bsl = iraf_sec_to_slice(self.header['BSEC1'], biassecoffsets)[::-1]
        tsl = iraf_sec_to_slice(self.header['TSEC1'])[::-1]

        db = self.data[bsl]
        rows = db.mean(axis=1)
        splx = np.arange(len(rows))
        spl = UnivariateSpline(splx, rows, s=splinesmoothing)
        splrows = spl(splx)

        sub = self.data - splrows.reshape(len(splrows), 1)

        result = sub[tsl]

        #now compute new uncertainty assuming poisson noise in electrons and
        #that the current `unc` is gaussian/read noise

        #g = self.header['GAIN']
        #unc = (result * g + (self.unc * g)**2)**0.5 / g
        unc = (result/self.header['GAIN'] + self.unc[tsl]**2)**0.5

        res = DoubleSpecImage(result, unc=unc, header=self.header,
                              side=self.side)
        pd = list(self.processingdone)
        pd.append('overscan sub')
        pd.append('overscan trim')
        res.processingdone = tuple(pd)
        return res

    def flatten(self, flatarr, flatunc=0):
        """
        Flatten the image with a normalized flat array as created by
        `create_normalized_flat`.

        Parameters
        ----------
        flatarr : array of size matching `data`
            The flat to divide this by
        flatunc : scalar or array of size matching `data`
            The uncertainty to assume for the flat (gaussian SD).

        Returns
        -------
        newimg : DoubleSpecImage
            The image after being flattened.
        """

        newdata = self.data / flatarr
        newunc = newdata * np.hypot(self.unc/self.data, flatunc/flatarr)

        res = DoubleSpecImage(newdata, unc=newunc, header=self.header,
                              side=self.side)
        pd = list(self.processingdone)
        pd.append('flattened')
        res.processingdone = tuple(pd)
        return res

    def extract_spectrum(self, aperture):
        """
        Extract a 1D spectrum the "simple" (non-Horne) way

        Parameters
        ----------
        aperture : 2-tuple
            The edges of the aperture along the spatial axis

        Returns
        -------
        spectrum : array
            A 1D extracted spectrum with the *sum* of the flux.
        unc : 1D array
            The flux uncertainty matching the extracted spectrum
        """

        apslc = aperture if isinstance(aperture, slice) else slice(*aperture)

        if self.side == 'red':
            d = self.data
        elif self.side == 'blue':
            d = self.data.T
        else:
            raise ValueError('unrecognized side ' + str(self.side))

        spec = np.sum(d[apslc], axis=0)
        unc = np.sum(d[apslc]**2, axis=0) ** 0.5

        return spec, unc

    def subtract_sky(self, skyapertures, skymodel=None):
        """
        Subtracts sky from the 2D image along the given apertures.

        Note that this currently assumes fitted sky has no uncertainty, which is
        probably only true with large enough sky apertures

        Parameters
        ----------
        skyapertures : list of 2-tuples or slice objects
            A list of 2-tuples (lower, upper) or slices, specifying the
            sections along the spatial direction to define the sky.
        skymodel : astropy.modeling.model
            The model to fit, or None to just subtract the mean

        Returns
        -------
        newimg : DoubleSpecImage
            A new 2D image with the sky subtracted
        fittedmodels : None or list of models
            The models fit along the row, or None if `skymodel` is None
        """
        from astropy import modeling

        apslcs = []
        for ap in skyapertures:
            if isinstance(ap, slice):
                apslcs.append(ap)
            else:
                apslcs.append(slice(*ap))

        if self.side == 'red':
            d = self.data.T  # first axis should be dispersion
        elif self.side == 'blue':
            d = self.data
        else:
            raise ValueError("unrecognized side " + str(self.side))

        subeds = []
        if skymodel:
            xs = np.arange(d.shape[1])
            xaps = np.concatenate([xs[ap] for ap in apslcs])

            if isinstance(skymodel, modeling.models.Linear1D):
                fitter = _linfit
            else:
                fitter = modeling.fitting.LevMarLSQFitter()#NonLinearLSQFitter()

            fitmodels = []
            for cut in d:
                yaps = np.concatenate([cut[ap] for ap in apslcs])
                fitmodels.append(fitter(skymodel, xaps, yaps))
                y = yaps
                subeds.append(cut - fitmodels[-1](xs))
        else:
            fitmodels = None
            for cut in d:
                yaps = np.concatenate([cut[ap] for ap in apslcs])
                subeds.append(cut - np.mean(yaps))

        # get the right spatial/dispersion axis convention
        if self.side == 'red':
            newdata = np.array(subeds).T
        elif self.side == 'blue':
            newdata = np.array(subeds)
        else:
            raise ValueError('unrecognized side '+str(self.side))

        res = DoubleSpecImage(newdata, self.unc, self.header,
                              self.side)
        pd = list(self.processingdone)
        pd.append('sky sub')
        res.processingdone = tuple(pd)

        return res, fitmodels

    def time_normalize(self):
        """
        Change to counts/sec instead of raw counts

        Returns
        -------
        newimg : DoubleSpecImage
            An image with value of counts/sec
        """
        newdata = self.data / self.header['EXPTIME']
        newunc = self.unc / self.header['EXPTIME']

        res = DoubleSpecImage(newdata, newunc, self.header, self.side)
        pd = list(self.processingdone)
        pd.append('time normalized')
        res.processingdone = tuple(pd)
        return res


def create_normalized_flat(images, imcombinefunc=np.median,
                           colcombinefunc=np.mean, pixsmooth=None):
    """
    Creates a normalized flat to apply to other Doublespec spectra.  Uses a
    smoothing spline

    Parameters
    ----------
    images : list of DoubleSpecImage
        The images to combine to make the flat
    imcombine : function
        The function to combine the images - should have an `axis` argument.
    imcombine : function
        The function to squash the image along the spatial axis - should have an
        `axis` argument.
    pixsmooth : int or None
        The number of pixels to smooth over for any given.  If negative or 0, no
        smoothing will happen. If None, will use the default of 30/10 for
        red/blue side.

    Returns
    -------
    normflat : array
        An array of size matching the images containing the normalized flat -
        i.e., the thing to divide by to flatten the images.
    """
    from warnings import warn
    from scipy import interpolate

    side = images[0].side
    for im in images:
        if im.side != side:
            raise ValueError('gave images from different sides - cannot flat!')
        if 'flat' not in im.objname.lower():
            warn('Object had name {0} which may not be a flat - are you sure '
                 'images are right?'.format(im.objname))

    combflat = imcombinefunc([im.data for im in images], axis=0)

    if side == 'red':
        #100:400 removes the edge effects where no light falls
        response = colcombinefunc(combflat[100:400], axis=0)
    elif side == 'blue':
        #100:340 removes the edge effects where no light falls
        response = colcombinefunc(combflat[:, 100:340], axis=1)
    else:
        raise ValueError('unrecognized side ' + str(side))

    if pixsmooth is None:
        if side == 'red':
            pixsmooth = 30
        elif side == 'blue':
            pixsmooth = 10
        else:
            raise ValueError("pixsmooth defaults don't know side " + str(side))

    if pixsmooth > 0:
        px = np.arange(len(response))
        ks = np.linspace(px[0], px[-1], len(px)/pixsmooth + 2)[1:-1]
        response_func = interpolate.LSQUnivariateSpline(px, response, t=ks)
        response = response_func(px)

    if side == 'red':
        normflat = combflat / response
    elif side == 'blue':
        normflat = combflat / response.reshape(len(response), 1)
    else:
        raise ValueError("flatting doesn't know side " + str(side))

    return normflat

def combine_imgs(imgs, combinetype='median'):
    """
    currently `combinetype` can only be 'median', 'mean', or 'sum'

    Note: need to dobule-check unk
    """
    datas = []
    uncs = []
    for img in imgs:
        datas.append(img.data)
        uncs.append(img.unc)

    if combinetype == 'median':
        newdata = np.median(datas, axis=0)
        newunc = (np.pi/2)**0.5 * np.sum(np.array(uncs)**2, axis=0)**0.5 / len(imgs)
    elif combinetype == 'mean':
        newdata = np.mean(datas, axis=0) / len(imgs)
        newunc = np.sum(np.array(uncs)**2, axis=0)**0.5 / len(imgs)
    elif combinetype=='sum':
        newdata = np.mean(datas, axis=0) / len(imgs)
        newunc = np.sum(np.array(uncs)**2, axis=0)**0.5
    else:
        raise ValueError('unrecognized combine type ' + str(combinetype))

    res = DoubleSpecImage(newdata, newunc, imgs[0].header, imgs[0].side)
    pd = []
    for img in imgs:
        for pdi in img.processingdone:
            if pdi not in pd:
                pd.append(pdi)
    pd.append(combinetype + ' combine')
    res.processingdone = tuple(pd)
    return res



class DispersionSolution(object):
    def __init__(self, arcimg, aperture, poly_order=1):
        self.arcimg = arcimg
        self.aperture = aperture
        self.poly_order = poly_order

        spec, unc = self.arcimg.extract_spectrum(aperture)
        self.spec1d = spec
        self.unc1d = unc

        self.linepixtowl = {}

    def pixtowl(self, pix):
        """
        pix to wl
        order==1 -> linear
        """
        if self.poly_order == 1:
            return self.linear_solution()(pix)
        else:
            return self.poly_solution(self.poly_order)(pix)

    def wltopix(self, wl):
        l = self.linear_solution()
        linpix = (wl - l.intercept) / l.slope
        if self.poly_order == 1:
            return linpix
        else:
            from scipy.optimize import newton

            p = self.poly_solution(self.poly_order)

            #use Newton's method to invert.  Note that this needs an initial
            #guess, for which we use the linear solution.
            return newton(lambda x: p(x) - wl, linpix)

    def plot_spec(self, lines=True):
        from matplotlib import pyplot as plt

        x = np.arange(len(self.spec1d))
        res = plt.plot(x, self.spec1d, linestyle='steps-mid')
        plt.xlim(x[0], x[-1])

        if lines:
            for px in self.linepixtowl:
                plt.axvline(px, color='r')

        return res

    def plot_spec_wl(self, lines=True, referenceatlas=None, markresidual=False):
        from matplotlib import pyplot as plt
        from matplotlib.transforms import blended_transform_factory

        x = np.arange(len(self.spec1d))
        wl = self.pixtowl(x)
        res = plt.plot(wl, self.spec1d, linestyle='steps-mid')
        plt.xlim(min(wl), max(wl))

        if referenceatlas:
            yl = plt.ylim()
            atlwl, atlspec = referenceatlas
            #rescale based on peak
            peakatl = atlspec.max()
            plt.plot(atlwl, atlspec*(yl[-1]/peakatl), linestyle='steps-mid', c='g')
            plt.ylim(*yl)

        if lines:
            xdatayax_trans = blended_transform_factory(plt.gca().transData,
                                                       plt.gca().transAxes)
            for px, wl in self.linepixtowl.items():
                plt.axvline(wl, color='r')
                if markresidual:
                    residual = wl - self.pixtowl(px)
                    plt.annotate('{0:.1f}'.format(residual), (float(wl), 0.1), transform=xdatayax_trans)

        return res

    def guess_line_loc(self, peakguess, truewl, minpeakratio, addtosoln=True):
        cen, cenval, peakmax, loweri, upperi, peakzonebase = self.peak_stats(peakguess)

        if minpeakratio is not None and (peakmax/peakzonebase) < minpeakratio:
            raise ValueError("Guessed line's peak to base ratio is {0}, which "
                "is less than {1}".format(peakmax/peakzonebase, minpeakratio))
        
        if addtosoln:
            self.linepixtowl[cen] = truewl
        return cen, truewl

    def linear_solution(self):
        from astropy.modeling.models import Linear1D

        m = Linear1D(1, 0)
        if len(self.linepixtowl) > 1:
            return _linfit(m, list(self.linepixtowl.keys()), list(self.linepixtowl.values()))
        else:
            raise ValueError('cannot make linear solution w < 2 points')

    def poly_solution(self, order):
        from astropy.modeling.fitting import LinearLSQFitter
        from astropy.modeling.models import Polynomial1D

        poly = Polynomial1D(order)
        ftr = LinearLSQFitter()
        poly = ftr(poly, list(self.linepixtowl.keys()), list(self.linepixtowl.values()))
        return poly

    def sigma_clip(self, siglimit=3, iters=None):
        """
        Iteratively sigma-clips lines that are `siglimit` standard deviations
        from the solution.

        Parameters
        ----------
        siglimit : float
            The number of standard deviations to clip
        iters : int or None
            The number of iterations to run, or None means iterate until
            converged

        Returns
        -------
        removed : dict
            A dictionary of pix to wl of the lines that were removed
        niters : int
            The number of iterations completed at convergence
        """
        removed = {}
        if iters is None:
            iters = np.inf

        i = 0
        while i < iters:
            x = np.array(list(self.linepixtowl.keys()))
            y = np.array(list(self.linepixtowl.values()))
            res = y - self.pixtowl(x)

            bad = np.abs(res) >= siglimit * np.std(res)

            if bad.sum() == 0:
                break

            for xval in x[bad]:
                removed[xval] = self.linepixtowl.pop(xval)

            i += 1

        return removed, i


    def peak_stats(self, peakguess):
        """
        Given a guess for a peak, return statistics about it

        Parameters
        ----------
        peakguess : int
            A guess at the location of a peak in index numbers

        Returns
        -------
        cen : float
            Centroid of the peak
        cenval : float
            Value at the centroid
        peakmax : float
            Maximum value of the peak
        loweridx : int
            lower index of the peak zone
        upperidx : int
            lower index of the peak zone
        peakzonebase : float
            Average of the values at ``loweridx-1`` and ``upperidx+1``
        """
        from collections import namedtuple

        deriv2_spec = getattr(self, '_deriv2_spec', None)
        if deriv2_spec is None:
            deriv2_spec = np.concatenate(([0], np.diff(self.spec1d, 2), [0]))
            self._deriv2_spec = deriv2_spec
            self._peakzone = peakzone = deriv2_spec < 0
        else:
            peakzone = self._peakzone

        x = np.arange(len(deriv2_spec))
        idx = int(round(peakguess))
        if idx>(len(peakzone) - 1) or idx<0:
            raise ValueError('requested peak location is not in the spectrum')

        if not peakzone[idx]:
            ok = False
            if idx+1 < len(peakzone) and idx-1 >= 0:
                if peakzone[idx+1] and not peakzone[idx-1]:
                    idx = idx+1
                    ok = True
                elif not peakzone[idx+1] and peakzone[idx-1]:
                    idx = idx-1
                    ok = True
            if not ok:
                raise ValueError("Tried to find peak at {0} but it's in a "
                                 "valley, not a peak".format(peakguess))

        offset = np.abs(idx - x)
        offset[peakzone] = len(deriv2_spec) + 1
        loweri = np.argmin(offset[:idx])
        offset[:idx] = len(deriv2_spec) + 1
        upperi = np.argmin(offset)
        peakslice = slice(loweri - 1, upperi + 1)

        xsliced = x[peakslice]
        specsliced = self.spec1d[peakslice]

        cen = sum(specsliced * xsliced) / specsliced.sum()
        cenval = np.interp(cen, xsliced, specsliced)
        peakmax = max(self.spec1d[loweri:upperi])
        peakzonebase = (self.spec1d[loweri-1] + self.spec1d[upperi+1]) / 2


        stats_tuple_type = namedtuple('peak_stats',
            'cen,cenval,peakmax,loweri,upperi,peakzonebase'.split(','))

        ld = locals()

        return stats_tuple_type(*[ld[nm] for nm in stats_tuple_type._fields])

    def guess_from_line_list(self, lst, verbose=True, minpeakratio=3,
                                   continuous_fit=True, sigmaclip=False,
                                   min_wl=0, max_wl=np.inf):
        """
        `lst` should be  IRAF-style line list entries
        """
        if sigmaclip is True:
            sigmaclip = {}
        elif not sigmaclip:
            sigmaclip = None

        if isinstance(lst, six.string_types):
            with open(lst) as f:
                lst = []
                for l in f:
                    lst.append(l)

        foundlines = []
        nfailedlines = 0
        for l in lst:
            ls = l.split()
            wl = float(ls[0])
            if wl < min_wl or wl > max_wl:
                print('Skipping line {0} at wl={1} - out of wl min/max'.format(ls[1], wl))
                continue
            try:
                result = self.guess_line_loc(self.wltopix(wl), wl,
                                             minpeakratio=minpeakratio,
                                             addtosoln=continuous_fit)
                foundlines.append(result)
                if verbose:
                    print('Found line {0} at wl={1}'.format(ls[1], wl))

                if sigmaclip is not None:
                    nclipped = len(self.sigma_clip(**sigmaclip)[0])
                    if verbose:
                        print('Clipped {0} lines'.format(nclipped))

            except ValueError as e:
                if verbose:
                    print('Failed to find line {0} at wl={1}: "{2}"'.format(ls[1], wl, e))
                nfailedlines += 1

        if not continuous_fit:
            for pix, wl in foundlines:
                self.linepixtowl[pix] = wl
            if sigmaclip is not None:
                self.sigma_clip(**sigmaclip)

        return len(foundlines), nfailedlines

    def plot_solution(self, residuals=False, sckwargs={}, plkwargs={}, plotsdres=True):
        from matplotlib import pyplot as plt

        x = np.arange(len(self.spec1d))

        sckwargs.setdefault('c', 'r')
        sckwargs.setdefault('lw', 0)
        plkwargs.setdefault('ls', '--')

        if residuals:
            lx = np.array(list(self.linepixtowl.keys()))
            ly = np.array(list(self.linepixtowl.values()))

            if residuals == 'linear':
                model = self.linear_solution()(lx)
            else:
                model = self.pixtowl(lx)
            res = ly - model
            stdres = np.std(res)

            plt.scatter(lx, res, **sckwargs)
            plt.plot((min(x), max(x)), [0, 0], **plkwargs)
            if plotsdres:
                plt.plot((min(x), max(x)), [stdres, stdres], c='k', lw=1, ls=':')
                plt.plot((min(x), max(x)), [-stdres, -stdres], c='k', lw=1, ls=':')
            plt.xlabel('pix')
            plt.ylabel('residuals [$\\AA$]')
        else:
            plt.scatter(list(self.linepixtowl.keys()), list(self.linepixtowl.values()), **sckwargs)
            plt.plot(x, self.pixtowl(x), **plkwargs)
            plt.xlabel('pix')
            plt.ylabel('$\\AA$')
        plt.xlim(min(x), max(x))


def find_all_obj(objnm, globpattern='*.fits'):
    from glob import glob
    from astropy.io import fits

    fns = glob(globpattern)
    return [fn for fn in fns if objnm == fits.getheader(fn)['OBJECT']]

def list_all_objs(globpattern='*.fits'):
    from glob import glob
    from astropy.io import fits

    fns = glob(globpattern)
    return [(fn, fits.getheader(fn)['OBJECT']) for fn in fns]


def nearest_in_line_list(wl, lst):
    wls = np.array([l.split()[0] for l in lst], dtype=float)
    idx = np.argmin(np.abs(wl - wls))
    return wls[idx], lst[idx]

#line lists from NOAO spectral atlas
FEAR_LINE_LIST="""
3020.6391  FeI
3024.0325  FeI
3037.3887  FeI
3047.6043  FeI
3057.4456  FeI
3059.0856  FeI
3225.7850  FeI
3243.6887  ArII
3350.9243  ArII
3376.4359  ArII
3388.5309  ArII
3399.3335  FeI
3404.3535  FeI
3407.4585  FeI
3413.1312  FeI
3427.1192  FeI
3443.8762  FeI
3465.8603  FeI
3475.4500  FeI
3476.7016  FeI
3480.5055  ArII
3490.5737  FeI
3509.7785  ArII
3519.9936  ArII
3535.3196  ArII
3541.0832  FeI
3548.5144  ArII
3559.5081  ArII
3561.0304  ArII
3565.3786  FeI
3576.6156  ArII
3581.1925  FeI
3588.4407  ArII
3608.8587  FeI
3618.7676  FeI
3631.4629  FeI
3647.8424  FeI
3679.9132  FeI
3687.4564  FeI
3705.5657  FeI
3709.2459  FeI
3718.2065  ArII
3719.9346  FeI
3722.5625  FeI
3727.6187  FeI
3729.3087  ArII
3733.3169  FeI
3734.8636  FeI
3737.1313  FeI
3745.5608  FeI
3748.2617  FeI
3749.4847  FeI
3758.2324  FeI
3763.7885  FeI
3767.1914  FeI
3812.9641  FeI
3815.8397  FeI
3820.4251  FeI
3824.4436  FeI
3825.8805  FeI
3827.8226  FeI
3834.2222  FeI
3856.3717  FeI
3859.9114  FeI
3868.5284  ArII
3878.5730  FeI
3886.2820  FeI
3895.6558  FeI
3899.7073  FeI
3902.9452  FeI
3906.4794  FeI
3920.2577  FeI
3922.9115  FeI
3925.7188  ArII
3930.2962  FeI
3946.0971  ArII
3948.9789  ArI
3969.2570  FeI
3979.3559  ArII
3994.7918  ArII
4005.2414  FeI
4033.8093  ArII
4042.8937  ArII
4044.4179  ArI
4045.8130  FeI
4052.9208  ArII
4063.5939  FeI
4103.9121  ArII
4118.5442  FeI
4131.7235  ArII
4143.8688  FeI
4158.5905  ArI
4164.1795  ArI
4198.3036  FeI
4200.6745  ArI
4216.1828  FeI
4222.6373  FeI
4237.2198  ArII
4259.3619  ArI
4271.7593  FeI
4277.5282  ArII
4294.1243  FeI
4300.1008  ArI
4307.9015  FeI
4315.0837  FeI
4325.7615  FeI
4331.1995  ArII
4333.5612  ArI
4335.3379  ArI
4337.0708  ArII
4345.1680  ArII
4348.0640  ArII
4352.2049  ArII
4362.0662  ArII
4367.8316  ArII
4375.9294  FeI
4379.6668  ArII
4383.5445  FeI
4385.0566  ArII
4400.9863  ArII
4404.7499  FeI
4415.1222  FeI
4426.0011  ArII
4427.3039  FeI
4430.1890  ArII
4433.8380  ArII
4439.4614  ArII
4448.8792  ArII
4461.6521  FeI
4466.5508  FeI
4474.7594  ArII
4481.8107  ArII
4489.7389  FeI
4490.9816  ArII
4498.5384  ArII
4502.9268  ArII
4510.7332  ArI
4522.3230  ArI
4528.6133  FeI
4530.5523  FeI
4545.0519  ArII
4579.3495  ArII
4589.8978  ArII
4596.0967  ArI
4598.7627  ArII
4609.5673  ArII
4628.4409  ArI
4637.2328  ArII
4647.4329  FeI
4657.9012  ArII
4702.3161  ArI
4726.8683  ArII
4732.0532  ArII
4735.9058  ArII
4764.8646  ArII
4806.0205  ArII
4847.8095  ArII
4859.7406  FeI
4879.8635  ArII
4889.0422  ArII
4891.4919  FeI
4920.5018  FeI
4933.2091  ArII
4957.5966  FeI
4965.0795  ArII
4972.1597  ArII
5006.1175  FeI
5009.3344  ArII
5012.0674  FeI
5017.1628  ArII
5051.6336  FeI
5062.0371  ArII
5083.3377  FeI
5090.4951  ArII
5110.3849  FeI
5125.7654  ArII
5141.7827  ArII
5145.3083  ArII
5162.2846  ArI
5167.4873  FeI
5171.5953  FeI
5187.7462  ArI
5194.9412  FeI
5221.2710  ArI
5227.1697  FeI
5232.9394  FeI
5266.5546  FeI
5269.5366  FeI
5283.6206  FeI
5302.2989  FeI
5324.1782  FeI
5328.0376  FeI
5341.0233  FeI
5371.4892  FeI
5397.1269  FeI
5405.7741  FeI
5415.1997  FeI
5421.3517  ArI
5424.0686  FeI
5429.6955  FeI
5434.5228  FeI
5439.9891  ArI
5446.8937  FeI
5451.6520  ArI
5455.6090  FeI
5495.8738  ArI
5514.3760  ArI
5524.9570  ArI
5558.7020  ArI
5569.6177  FeI
5577.6845  ArII
5586.7553  FeI
5606.7330  ArI
5615.6436  FeI
5648.6863  ArI
5650.7043  ArI
5681.9001  ArI
5691.6612  ArII
5739.5196  ArI
5783.5360  ArI
5786.5553  ArII
5802.0798  ArI
5812.7592  ArII
5834.2633  ArI
5860.3103  ArI
5882.6242  ArI
5888.5841  ArI
5912.0853  ArI
5916.5992  ArI
5927.1258  ArI
5928.8130  ArI
5942.6686  ArI
5949.2583  ArI
5964.4723  ArI
5968.3199  ArI
5971.6008  ArI
5987.3016  ArI
5998.9987  ArI
6005.7242  ArI
6013.6777  ArI
6025.1500  ArI
6032.1274  ArI
6043.2233  ArI
6046.8977  ArI
6052.7229  ArI
6059.3725  ArI
6081.2433  ArI
6085.8797  ArI
6090.7848  ArI
6098.8031  ArI
6105.6351  ArI
6114.9234  ArII
6123.3619  ArII
6127.4160  ArI
6145.4411  ArI
6155.2385  ArI
6165.1232  ArI
6170.1740  ArII
6172.2778  FeI
6191.5583  ArII
6201.1002  ArI
6212.5031  ArI
6215.9383  FeI
6230.7260  ArII
6243.1201  FeI
6246.3172  ArI
6248.4055  FeI
6252.5537  FeI
6296.8722  ArI
6307.6570  ArII
6324.4163  ArI
6364.8937  ArI
6369.5748  ArI
6384.7169  FeI
6399.9995  ArII
6403.0128  FeI
6411.6468  ArI
6416.3071  ArII
6437.6003  ArI
6466.5526  ArII
6483.0825  ArII
6538.1120  ArI
6604.8534  ArI
6614.3475  ArII
6620.9665  ArII
6632.0837  ArI
6638.2207  ArII
6639.7403  ArII
6643.6976  ArII
6656.9386  ArI
6660.6761  ArI
6664.0510  ArI
6666.3588  ArII
6677.2817  ArI
6684.2929  ArII
6719.2184  ArI
6752.8335  ArI
6766.6117  ArI
6861.2688  ArII
6871.2891  ArI
6879.5824  ArI
6888.1742  ArI
6937.6642  ArI
6951.4776  ArI
6965.4307  ArI
7030.2514  ArI
7067.2181  ArI
7107.4778  ArI
7125.8200  ArI
7147.0416  ArI
7158.8387  ArI
7206.9804  ArI
7265.1724  ArI
7272.9359  ArI
7311.7159  ArI
7316.0050  ArI
7353.2930  ArI
7372.1184  ArI
7383.9805  ArI
7392.9801  ArI
7412.3368  ArI
7425.2942  ArI
7435.3683  ArI
7471.1641  ArI
7484.3267  ArI
7503.8691  ArI
7514.6518  ArI
7589.3151  ArII
7635.1060  ArI
7670.0575  ArI
7798.5604  ArI
7868.1946  ArI
7891.0750  ArI
7916.4420  ArI
7948.1764  ArI
7948.1964  ArI
8006.1567  ArI
8014.7857  ArI
8037.2183  ArI
8046.1169  ArI
8053.3085  ArI
8066.6046  ArI
8103.6931  ArI
8115.3110  ArI
8143.5050  ArI
8203.4352  ArI
8264.5225  ArI
8327.0526  FeI
8384.7240  ArI
8387.7700  FeI
8408.2096  ArI
8424.6475  ArI
8490.3065  ArI
8521.4422  ArI
8605.7762  ArI
8620.4602  ArI
8667.9442  ArI
8688.6213  FeI
8761.6862  ArI
8799.0875  ArI
8962.1468  ArI
9008.4636  ArII
9017.5912  ArII
9122.9674  ArI
9194.6385  ArI
9224.4992  ArI
9291.5313  ArI
9354.2198  ArI
9508.4513  ArI
9657.7863  ArI
9657.7863  ArI
9784.5028  ArI
10470.0535  ArI
"""[1:-1].split('\n')

HENEAR_LINE_LIST="""
3187.745     HeI
3307.2283    ArII
3319.3446    ArI
3350.9243    ArII
3354.55      HeI
3373.4823    ArI
3376.4359    ArII
3388.5309    ArII
3397.866     NeII
3406.1804    ArI
3414.4583    ArII
3417.9035    NeI
3421.6107    ArII
3454.0952    ArII
3464.1272    ArII
3472.5711    NeI
3476.7474    ArII
3480.5055    ArII
3509.7785    ArII
3514.3877    ArII
3535.3196    ArII
3548.5144    ArII
3554.3058    ArI
3559.5081    ArII
3561.0304    ArII
3567.6564    ArI
3572.296     ArI
3576.6156    ArII
3588.4407    ArII
3600.1691    NeI
3606.5218    ArI
3613.643     HeI
3622.1375    ArII
3637.031     ArII
3639.8329    ArII
3643.1169    ArI
3655.2782    ArII
3690.8951    ArI
3718.2065    ArII
3720.4265    ArII
3724.5165    ArII
3729.3087    ArII
3737.889     ArII
3780.8398    ArII
3786.3824    ArII
3796.5934    ArII
3799.382     ArII
3803.1724    ArII
3809.4561    ArII
3819.6072    HeI
3834.6787    ArI
3850.5813    ArII
3868.5284    ArII
3875.2645    ArII
3888.648     HeI
3914.7675    ArII
3925.7188    ArII
3928.6233    ArII
3932.5466    ArII
3946.0971    ArII
3947.5046    ArI
3948.9789    ArI
3964.7289    HeI
3968.3594    ArII
3974.4766    ArII
3979.3559    ArII
3994.7918    ArII
4009.268     HeI
4013.8566    ArII
4033.8093    ArII
4035.46      ArII
4044.4179    ArI
4052.9208    ArII
4054.5258    ArI
4082.3872    ArII
4103.9121    ArII
4112.8153    ArII
4120.815     HeI
4131.7235    ArII
4156.086     ArII
4158.5907    ArI
4164.1795    ArI
4168.967     HeI
4181.8836    ArI
4198.317     ArI
4200.6745    ArI
4222.6373    ArII
4237.2198    ArII
4251.1846    ArI
4259.3619    ArI
4272.1689    ArI
4277.5282    ArII
4282.8976    ArII
4300.1008    ArI
4331.1995    ArII
4333.5612    ArI
4335.3379    ArI
4345.168     ArI
4348.064     ArII
4352.2049    ArII
4367.8316    ArII
4379.6668    ArII
4426.0011    ArII
4430.189     ArII
4437.551     HeI
4448.8792    ArII
4453.9177    KrI
4463.6901    KrI
4471.479     HeI
4474.7594    ArII
4481.8107    ArII
4490.9816    ArII
4498.5384    ArII
4502.3546    KrI
4510.7332    ArI
4522.323     ArI
4530.5523    ArII
4535.4903    ArII
4537.6426    ArII
4545.0519    ArII
4579.3495    ArII
4589.8978    ArII
4596.0967    ArI
4598.7627    ArII
4609.5673    ArII
4628.4409    ArI
4637.2328    ArII
4657.9012    ArII
4702.3161    ArI
4713.1455    HeI
4721.591     ArII
4726.8683    ArII
4735.9058    ArII
4764.8646    ArII
4768.675     ArI
4806.0205    ArII
4847.8095    ArII
4865.91      ArII
4876.2611    ArI
4879.8635    ArII
4894.6909    ArI
4904.7516    ArII
4921.931     HeI
4933.2091    ArII
4965.0795    ArII
4972.1597    ArII
5009.3344    ArII
5015.6779    HeI
5047.738     HeI
5062.0371    ArII
5141.7827    ArII
5145.3083    ArII
5151.3907    ArI
5162.2846    ArI
5187.7462    ArI
5221.271     ArI
5252.788     ArI
5330.7775    NeI
5373.4943    ArI
5393.9719    ArI
5400.5617    NeI
5410.473     ArI
5421.3517    ArI
5451.652     ArI
5467.1704    ArI
5473.4516    ArI
5495.8738    ArI
5506.1128    ArI
5524.957     ArI
5588.72      ArI
5597.4756    ArI
5606.733     ArI
5681.9001    ArI
5689.8163    NeI
5700.873     ArI
5739.5196    ArI
5748.2985    NeI
5764.4188    NeI
5772.1143    ArI
5783.536     ArI
5820.1558    NeI
5852.4878    NeI
5875.621     HeI
5881.895     NeI
5888.5841    ArI
5912.0853    ArI
5928.813     ArI
5944.8342    NeI
5975.534     NeI
6043.2233    ArI
6074.3377    NeI
6096.1631    NeI
6114.9234    ArII
6128.4499    NeI
6143.0626    NeI
6155.2385    ArI
6163.5939    NeI
6212.5031    ArI
6217.2812    NeI
6266.495     NeI
6296.8722    ArI
6304.789     NeI
6334.4278    NeI
6382.9917    NeI
6402.246     NeI
6416.3071    ArI
6456.291     KrI
6506.5281    NeI
6532.8822    NeI
6598.9529    NeI
6717.043     NeI
6752.8335    ArI
6766.6117    ArI
6871.2891    ArI
6929.4673    NeI
6937.6642    ArI
6965.4307    ArI
7024.0504    NeI
7032.4131    NeI
7107.4778    ArI
7125.82      ArI
7147.0416    ArI
7173.9381    NeI
7206.9804    ArI
7245.1666    NeI
7272.9359    ArI
7281.349     HeI
7353.293     ArI
7372.1184    ArI
7383.9805    ArI
7488.8712    NeI
7503.8691    ArI
7514.6518    ArI
7535.7739    NeI
7544.0443    NeI
7587.413     KrI
7601.5443    KrI
7635.106     ArI
7685.246     KrI
7694.5393    KrI
7854.8215    KrI
7891.075     ArI
7913.4242    KrI
7948.1764    ArI
8006.1567    ArI
8014.7857    ArI
8059.5038    KrI
8103.6931    ArI
8115.311     ArI
8190.0543    KrI
8264.5225    ArI
8281.0495    KrI
8377.6065    NeI
8408.2096    ArI
8424.6475    ArI
8495.3598    NeI
8508.87      KrI
8521.4422    ArI
8591.2583    NeI
8605.7762    ArI
8620.4602    ArI
8634.647     NeI
8654.3831    NeI
8667.9442    ArI
8776.749     KrI
8799.0875    ArI
8919.5007    NeI
8928.692     KrI
9122.9674    ArI
9194.6385    ArI
9224.4992    ArI
9291.5313    ArI
9354.2198    ArI
9657.7863    ArI
9751.759     KrI
9784.5028    ArI
9856.24      KrI
10052.1      ArI
10254.04     ArI
10309.15     ArI
10332.76     ArI
10360.37     KrI
10470.054    ArI
10506.47     ArI
10529.32     ArI
10593.01     KrI
10673.55     ArI
10681.78     ArI
10712.77     ArI
10733.87     ArI
10759.13     ArI
10773.35     ArI
10798.12     NeI
10830.337    HeI
10844.54     NeI
10880.96     ArI
10950.74     ArI
11078.87     ArI
11106.44     ArI
"""[1:-1].split('\n')
HENEAR_NOKR_LINE_LIST = [l for l in HENEAR_LINE_LIST if 'Kr' not in l]
