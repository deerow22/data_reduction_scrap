
from matplotlib import pyplot as plt

import os

import numpy as np

from glob import glob

import reduce_dblspec 

from astropy.io import fits
from astropy import units as u
from astropy import modeling
from astropy.table import Table

from astropy import modeling
from astropy import constants as cnst


def guess_plot(pxguess, wlguess, specset, pxwindw=300, wlwindow=400): 
    fig, (ax1, ax2) = plt.subplots(1,2)

    ds, wlatlas, specatlas, linelist = specset 
    
    #use these graphs to refine matching of arcs to atlas
    
    #first graph 
    plt.axes(ax1)
    ds.plot_spec() #1D spectrum of dispersion solution or is this the arc..waiting on eriks answer
    plt.axvline(pxguess, color='k', ls=':') #plot a vertical line where think peak pixel is in emission lines
    plt.xlim(pxguess-pxwindw/2, pxguess+pxwindw/2)
    #plt.ylim(0, np.max(ds.spec1d[plt.xlim()[0] : plt.xlim()[1]])*1.05)
    plt.title('DS 1D Spectrum')

    
    #second graph
    plt.axes(ax2)
    plt.plot(wlatlas, specatlas) #plots what atlas spectrum looks like
    plt.title('Atlas 1D Spectrum')
    plt.axvline(wlguess, color='k', ls=':')
    plt.xlim(wlguess-wlwindow/2, wlguess+wlwindow/2)
    
    #defs the min & max wavelength peaks to use to scale in next lines
    minpx = np.argmin(np.abs(wlatlas-(wlguess-wlwindow/2)))
    maxpx = np.argmin(np.abs(wlatlas-(wlguess+wlwindow/2)))
    
  
    
    #sets wavelength scales based on graph size-ish
    plt.ylim(0, specatlas[minpx:maxpx].max()*1.05)
    guess = reduce_dblspec.nearest_in_line_list(wlguess, linelist)
    plt.axvline(guess[0], color='r', ls=':') #for 1st wavelength to linelist match draw red vertical line
    
    return guess #outputs wavelength guesses based on atlas spec known linelists
    
    
def dispersion(all_side_files,atlasfiles,arcfiles,linelist,flatid):
    #Overscan the flats
    flats = [reduce_dblspec.DoubleSpecImage(fn).overscan_sub_and_trim() for fn in 
          reduce_dblspec.find_all_obj(flatid,all_side_files)[:10]] #DO I WANNA KEEP AS ONLY LAST 10???
          #flatid is the name you gave your flats (need'') ex: 'DomeFlat'
    
    #small scale flat creationg - normalized
    nfl = reduce_dblspec.create_normalized_flat(flats)

    header = atlasfiles[0].header    
    specnoao = atlasfiles[0].data
    wlnoao = header['CRVAL1'] + header['CD1_1']*(np.arange(len(specnoao)) - header['CRPIX1'] + 1) #+1 is for 0-based

    #creating dispersion solution
    ds = reduce_dblspec.DispersionSolution(reduce_dblspec.DoubleSpecImage(arcfiles).overscan_sub_and_trim().flatten(nfl), (320, 360))
    
    #print(header['IRAFNAME']) 
    
    #outputs graphs of atlas
    plt.figure(figsize=(20,5))
    plt.plot(wlnoao, specnoao)
    plt.title('Atlas 1D Spectrum-wavelengths')
    plt.xlabel('wavelengths')
    #atlas plot axis range limit choices
    if header['IRAFNAME']=='henear.spec': #redside
        #print('red') #tested & worked
        print('used atlas plot limits for Red side')
        plt.xlim(5000,10000)
    elif header['IRAFNAME']=='FeAr.spec': #blueside
        #print('blue') #tested & worked
        print('used atlas plot limits for Blue side')
        plt.xlim(3000,5500)
        plt.ylim(0,5e5) #how to make this more flexible?
    else: #not sure why this prints for redside b/c its still choosing the correct xrange&blue does too
        print("can't determine camera side for atlas plot limits based on:",header['IRAFNAME'])
        #raise ValueError("Can't determine camera side ")    
    
    #outputting arc graph so can guess lines
    plt.figure(figsize=(20,5))
    plt.xlabel('pixels')
    plt.title('Arc Lamp 1D Spectrum-pixels')
    ds.plot_spec()
    #extra condition just for blue side orientation b/c of camera inversion to make comparison to atlas easier
    if header['IRAFNAME']=='FeAr.spec': #blueside
        plt.xlim([2800,0]) #for arc
    
    specset = (ds, wlnoao, specnoao,linelist)
    return (specset) #returns input for guess_plot along with graphs to make pixel & wl guesses
    
    
def goodness_of_matches(arcfiles, guesses, all_side_files, atlasfiles, linelist, flatid):
    flats = [reduce_dblspec.DoubleSpecImage(fn).overscan_sub_and_trim() for fn in 
          reduce_dblspec.find_all_obj(flatid,all_side_files)[:10]] #DO I WANNA KEEP AS ONLY LAST 10???
          #flatid is how you named your flat files, ex: 'DomeFlat'
    #outputs = []
    #small scale flat creationg - normalized
    nfl = reduce_dblspec.create_normalized_flat(flats)
    header = atlasfiles[0].header    
    
    arcimg = reduce_dblspec.DoubleSpecImage(arcfiles).overscan_sub_and_trim().flatten(nfl)
    
    #print(header['IRAFNAME'])
    if header['IRAFNAME']=='henear.spec': #redside
        print('chose ds for redside')
        ds = reduce_dblspec.DispersionSolution(arcimg, (320, 360), poly_order=3)
        #print('ds for red')
    elif header['IRAFNAME']=='FeAr.spec': #blueside
        print('chose ds for blueside')
        ds = reduce_dblspec.DispersionSolution(arcimg, (205, 260), poly_order=3)
    else:#for some reason this prints AS WELL AS prints indicating red side choosen for red side only, blue works fine
        print("camera side for ds not understood based on:",header['IRAFNAME'])
        #raise ValueError("Can't determine camera side ")


    
    outputs = []
    for i in guesses:
            list_a = ds.guess_line_loc(i[0], i[1],minpeakratio=2.5) #guesses(pixel, wl)
            outputs.append(list_a) #why am i collecting these? where are they used?
            
    #print(header['IRAFNAME'])
    if header['IRAFNAME']=='FeAr.spec': #blueside
        print('chose guesses for blueside')
        ds.guess_from_line_list(linelist,minpeakratio=5, continuous_fit=True, sigmaclip=True, max_wl=5400)    
    elif header['IRAFNAME']=='henear.spec': #redside
        print('chose guesses for redside')
        ds.guess_from_line_list(linelist,minpeakratio=10, continuous_fit=True, sigmaclip=True, min_wl=5600)
    else:
        print("camera side for guesses not understood based on:",header['IRAFNAME'])
        #raise ValueError("Can't determine camera side ")
        #ds.guess_from_line_list(linelist,minpeakratio=10, continuous_fit=True, sigmaclip=True, min_wl=3000)
    



    #1st graph-residuals
    plt.figure(figsize=(20,5))
    residuals = ds.plot_solution(True)  
    #2nd graph-populated matches, px to wl;want linear 
    plt.figure(figsize=(20,5))
    ds.plot_spec_wl()
    
    return (ds) 




    
#skyaps=[(lower bound pixel location in (x,y), upper bound pixel location in (x,y))]
#play with vmax & vmin to get better contrast
def sky_subtract(targetID,skyaps,ds,all_side_files,atlasfiles,all_side_files_target,flatid,specaps,vmax,vmin,model):
    #need these again -flats/overscan
    flats = [reduce_dblspec.DoubleSpecImage(fn).overscan_sub_and_trim() for fn in 
          reduce_dblspec.find_all_obj(flatid,all_side_files)[:10]] #10 b/c we took 10 flats
    nfl = reduce_dblspec.create_normalized_flat(flats)
    
    header = atlasfiles[0].header    


    
    #locating all files w/ the target & flat/overscan subt
    specs = [reduce_dblspec.DoubleSpecImage(fn).overscan_sub_and_trim().flatten(nfl) 
          for fn in reduce_dblspec.find_all_obj(targetID,all_side_files_target)]
    
    
    print(len(specs),"target files located")


    
    
    #stacking all that data for specific target-help increase s/n 
    comb = reduce_dblspec.combine_imgs(specs)
    
    
    
    
    
    #plotting 2D to get skyaps right around the target
    plt.figure(figsize=(20,10)) #setting up fig size
    value = model[1]


    if model == 'linear':
        subimg, models = comb.subtract_sky(skyaps, skymodel=modeling.models.Linear1D(1,0)) #(1,0)
    elif model == ('poly',value):
        subimg, models = comb.subtract_sky(skyaps, skymodel=modeling.models.Polynomial1D(value)) #Linear1D(1/10,10)) #(1,0)
    elif model == ('legendre',value):
        mymodel1=(modeling.polynomial.Legendre1D(value))
        subimg, models = comb.subtract_sky(skyaps, skymodel=mymodel1)
    elif model == ('hermite',value):
        mymodel2=(modeling.polynomial.Hermite1D(value))
        subimg, models = comb.subtract_sky(skyaps, skymodel=mymodel2)
    elif model == ('cheb',value):
        mymodel3=(modeling.polynomial.Chebyshev1D(value))
        subimg, models = comb.subtract_sky(skyaps, skymodel=mymodel3)
    else:
        print ("Model type or polynomial degree not recognized")
    

    plt.subplot(211)
    plt.title('2D spec - Verify skyaps for correct target extraction')
    if header['IRAFNAME']=='FeAr.spec': #blueside
        print('choose 2D spec for blueside')
        subimg.show_image(transpose=True,vmax=vmax,vmin=vmin) #replace as vmax=40,vmin=-10 if this doesnt work out well
        plt.figure(figsize=(20,10)) #setting up fig size
    elif header['IRAFNAME']=='henear.spec': #redside
        print('choose 2D spec for redside')
        subimg.show_image(transpose=False,vmax=vmax,vmin=vmin) #also replace here if doesnt work
        plt.figure(figsize=(20,10)) #setting up fig size
    else:
        print("camera side for 2D spec not understood based on:",header['IRAFNAME'])
        #raise ValueError("Can't determine camera side ")    
    
    
    #setting up 1D extraction of target
    #setting up extraction range based on sky apps not sure if these is good to do b/c depends on extension of gal size
    ##begin = (skyaps[0][0] + 60)
    ##print ('this is beginning value of target spec extraction:',begin)
    ##end = (skyaps[1][0] - 30)
    ##print('this is end value of target spec extraction:',end)
    ##flux, unc = subimg.extract_spectrum((begin,end))
    
    flux, unc = subimg.extract_spectrum(specaps)


    
    #plotting 1D extraction of target
    plt.subplot(212)
    plt.step(ds.pixtowl(np.arange(len(flux))), flux)
    plt.title('1D Target Spectrum')
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    #based below limit choices on info on palomar dblspec webpage
    if header['IRAFNAME']=='FeAr.spec': #blueside
        print('choose 1D target spec axis limits for blueside')
        plt.xlim(2800, 7000) #3600,5800 
        plt.ylim(0, plt.ylim()[-1])
    elif header['IRAFNAME']=='henear.spec': #redside
        print('choose 1D target spec axis limits for redside')
        plt.xlim(4700, 11000)
        plt.ylim(0, plt.ylim()[-1])

    
    return (flux, unc, comb)



#this returns the complete 1D flux array -except still need flux calibrations!
def combine_red_blue(ds_red,ds_blue,flux_red,flux_blue):
    #wl = np.append(ds_red.pixtowl(np.arange(len(flux_red))), ds_blue.pixtowl(np.arange(len(flux_blue))))
    #applying the dispersion solutions for side & slit size
    wl_red = ds_red.pixtowl(np.arange(len(flux_red)))
    wl_blue = ds_blue.pixtowl(np.arange(len(flux_blue)))
    
    #restructuring to work w/ interp & making cuts to avoid strange things at detector edges 
    wl_blue=wl_blue[::-1] #this reverses the array to become increasing order
    flux_blue=flux_blue[::-1] #need to also reverse this so associated vals agree w/ wl
    flux_red=flux_red[10::] #chops off extreme values near edges, seem to only be a prob on red side
    wl_red=wl_red[10::] #matched flux chop
    
    #combining wls & flux for both sides
    wl_both = np.concatenate([wl_red, wl_blue])
    wl_both = np.sort(wl_both)   
    rspec = np.interp(wl_both,wl_red,flux_red,left=0) #need to do so doesnt repeat first val
    bspec = np.interp(wl_both,wl_blue,flux_blue,right=0) #need so doesnt repeat last val
    rbflux = rspec + bspec #adding the whole specs so overlap sums and rest is only +0 so ok
    #print('rbflux?',rbflux)
    
    plt.step(wl_both, rbflux) #whats the diff between step & plot here? doesnt seem to be one
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.title('1D Spectrum for Target')
    plt.ylim(0, max(rbflux)) 
    print('this is the complete 1D flux array:')
    return(rbflux,wl_both)
