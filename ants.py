from netpython import pynet,netext
from scipy.misc import imread
import pylab
import numpy as np
from scipy.stats import pearsonr

from random import random
from random import shuffle
import csv

# ========================================================================================
#
#               code used for data analysis in
#               Schultner, Saramaki, Helantera:
#               Network analysis reveals complex genetic substructure in ant supercolonies
#
#               author: Jari Saramaki, jari.saramaki@aalto.fi
#               v1.0
#
# ========================================================================================

# ====== USAGE EXAMPLE ==============================================
#
# (in e.g. ipython shell)

# alltypes=["eworker","queen","lateworker","malepupa","queenpupa","workerpupa"] # all ant types in data
# antdict=ants.load_relatedness_data(filename="MYLA_allSamples_Genotypes.txt",colonyfilter='MY')  # reads data for the MY supercolony
# net_workers_MY=ants.antnet(antdict,anttypes=["eworker"],reftypes=alltypes) # generates a network for early workers in MY, using early workers and queens for background relateness
# coords = ants.readcoords()
# rshuffle_workers_MY=ants.shuffled_rvalues_conserve(antdict,100,anttypes=["eworker"],reftypes=alltypes) # generates 100 reference nets between early workers with ants randomly shuffled between nests
# ants.plot_map_network_pvalue(net_workers_MY,coords,rshuffle_workers_MY,show='MY')

# ======= PYNET NETWORK FORMAT =======================================
#
# The object net (a pynet.Symmnet object) returns the relatedness between
# sites i and j as follows: net[i][j]
# Note that we add 1.0 to relatedness values because a relatedness of 0.0
# would be interpreted as missing link. So subtract 1.0 whenever necessary.


# ====== GLOBALS ====================================================

# --- geocoordinate limits for supercolonies in MYLA_allSamples_Genotypes.txt

xlim_all=[285800,290000]
ylim_all=[6651400,6656200]

xlim_LA=[286100,286600]
ylim_LA=[6651790,6652160]

xlim_MY=[289200,289700]
ylim_MY=[6655480,6655870]

# --- REPLACE WITH YOUR WORKING PATH

mypath='/users/jsaramak/Documents/working/murkut/'

# ======= READING DATA AND CONSTRUCTING NETWORKS ==============================================

def load_relatedness_data(filename="MYLA_allSamples_Genotypes.txt",anttypes=["eworker","queen","lateworker","malepupa","queenpupa","workerpupa"],colonyfilter='all',firstlocus=4,lastlocus=12):

    '''Reads microsatellite data in the Relatedness format, and produces an antdict similarly to loadstruct_removeloci.
       Filename = path to relatedness-formatted file,
       anttypes = list of ant types to be taken into account,
       colonyfilter = "LA","MY", or "all",
       firstlocus,lastlocus = indices to first and last entries in file row containing data on loci'''

    antdict={}

    fn=open(mypath+filename,'rU')

    f=csv.reader(fn,delimiter='\t')

    # get rid of two header lines

    f.next() # no useful information

    h=f.next()

    loci=[locus for locus in h[firstlocus:lastlocus]] 

    while True:

        try:

            line=f.next()

            if (colonyfilter=='all' or line[1]==colonyfilter) and (line[3] in anttypes):

                tempdict={}
                
                ant=line[0]

                tempdict['colony']=line[1]

                tempdict['site']=line[2]

                tempdict['type']=line[3]

                tempdict['loci']=loci

                # next handle the alleles

                allele_list=[]

                for i in range(4,12):

                    if line[i]: # if data not missing:

                        if not(line[3]=='malepupa'):

                            alleles=line[i].split('/')

                            allele_list.append((int(alleles[0]),int(alleles[1])))

                        else:

                            allele=int(line[i])
                            allele_list.append((allele,allele))

                    else:

                        allele_list.append((-999,-999))

                tempdict['alleles']=allele_list

                antdict[ant]=tempdict
                
        except:

            break

    return antdict


def antnet(antdict,anttypes=['eworker','queen'],reftypes=['eworker','queen'],colonyavg=False,verbose=False):

    ''' Used to construct relatedness networks between ants of types listed in anttypes
     the chosen type should be listed in anttypes (use list even for single type, eg anttypes=["eworker"]
     reftypes include anttypes used in computing relatedness background (preferably all types)
     relatedness is calculated between i-j and j-i and the final r_{ij} is then averaged over these
     colonyavg: if True, background relatednesses calculated s.t. all colonies have the same contribution to the average
     independent of their N_ants; if very different sizes small colonies overweighted; better use False here

     NOTE: FOR VISUALIZATION AND PROPER WORKING OF NETWORKS, +1.0 IS ADDED TO EACH RELATEDNESS VALUE (r-values have offset +1.0!!)
     when analyzing relatedness values, ALWAYS SUBTRACT ONE!!! 
    
    '''

    antnet=pynet.SymmNet() # symmetric relatedness network, symmetric r12 will be average of asymmetric r12 and r21

    allele_counts,total_valid_loci=get_allele_counts(antdict,reftypes)
    tempsites=allele_counts.keys()

    # check if anttypes present at all sites; if not, do not include in network
    # (still use all you have for background relatedness ..)

    typecounts=typestats(antdict,anttypes=anttypes,verbose=False)

    sites=[]

    for site in tempsites:

        N_ants=0

        for anttype in anttypes:

            if anttype in typecounts[site]:

                N_ants+=typecounts[site][anttype]

        if N_ants>0:

            sites.append(site)

    ant_sitedict=get_ant_sitedict(antdict,anttypes)

    L=len(sites)

    for i in xrange(0,L-1):

        for j in xrange(i+1,L):

            site1=sites[i]
            site2=sites[j]

            # compute site1-site2 and site2-site1 relatednesses

            r12=r_populations(antdict,ant_sitedict,site1,site2,anttypes,reftypes,allele_counts,total_valid_loci,colonyavg=colonyavg)
            r21=r_populations(antdict,ant_sitedict,site2,site1,anttypes,reftypes,allele_counts,total_valid_loci,colonyavg=colonyavg)

            r=0.5*(r12+r21)

            antnet[site1][site2]=r+1.0 # +1 because Pynet.Symmnet doesn't put a link if r=0, so just to be very sure +1 added to all r values

    return antnet

def readcoords(coordfile='geocoords_all.csv'):

    '''Reads in a dictionary of geocoordinates for nest locations, in the Finnish coordinate system
       ETRS-TM35-FIN (EUREF-FIN) (WGS84)'''

    filename=mypath+coordfile

    cdict={}

    f=open(filename,'rU')

    for line in f:

        items=line.split(',')

        site=items[0].strip()

        c1=float(items[1].strip())

        c2=float(items[2].strip())

        cdict[site]=(c1,c2)

    return cdict

def relatedness_values(net,addone=True):

    '''Yields a list of pairwise relatedness values in a network'''

    rvalues=[e[2] for e in list(net.edges)]

    return rvalues

# ==================== PLOTTING AND VISUALIZATION ===================================================
#

def plot_map_network_pvalue(orignet,coords,rvalues,imgfile='K344.tif',pvalue=0.05,nodesize=10.0,notitle=True,bonferroni=False,filename='temp',nodedict={},addone_links=True,addone_nodes=False,addone_refs=True,ext_scale=False,minvalue=0.0,maxvalue=0.0,labelson=True,smallnodes=False,smallinks=False,singlecolor=False,mycolor='k',show='all',alpha=1.0):

    '''Visualizes the relatedness network orignet using coordinates in dict coords, thresholding the network such that only links that have
       relatednesses with likelihoods below pvalue in shuffled reference data input in rvalues are shown.
       Input parameters:
       orignet - the network to be visualized
       coords - a dictionary of coordinates for nests
       rvalues - a list of reference relatedness values from shuffling, computed with ants.shuffled_rvalues_conserve
       imgfile - the background map image as .tif
       pvalue - sets the threshold, using rvalues s.t. probability of r>r_{threshold} is equal/smaller than p in rvalues. Only links with
                r over the threshold value are shown.
       notitle - show/don't show plot title (default False)
       bonferonni - apply Bonferonni correction to pvalue, default False
       filename - name of file to save to
       nodedict - dict where keys = nodes, values = e.g. within-nest relatednesses. Used in coloring nodes.
       addone_links,addone_nodes,addone_refs = True. Use True when and if the 1.0 bias has been added to relatedness values.
       ext_scale - (True/False) use/don't use an externally defined color scale for links. (False: use min and max r of your network)
       min_value, max_value - min and max r for the above ext_scale
       show = ('MY'|'LA'|'all') - displays area covering MY or LA or both.'''

    if not(ext_scale):

        minr,maxr,meanr=get_min_max_r([orignet]) # ORIGINAL NETWORK'S VALUES FOR WEIGHTS ETC

        minr=minr-1.0
        maxr=maxr-1.0

    else:

        minr=minvalue # USE EXTERNAL LIMITS TO SET SCALES FOR NODE AND LINK COLORS
        maxr=maxvalue # IF EXTERNAL LIMITS USED THESE SHOULD NOT CONTAIN OFFSET!!!

    if not(pvalue==-999):

        tempnet,threshold=threshold_by_pvalue(orignet,rvalues,pvalue=pvalue,bonferroni=bonferroni,orig_addone=addone_links,ref_addone=addone_refs)

    else:

        tempnet=orignet

    if addone_links:

        net=net_subtract_one(tempnet)

    else:

        net=tempnet

    #min_value=min(nodedict.values())
    #max_value=max(nodedict.values())

    if not(pvalue==-999):

        titlestring=filename+', threshold at p=%1.2f'%(pvalue)+' r=%1.2f'%(threshold)

    else:

        titlestring=filename+', no threshold'

    if notitle:

        titlestring=''

    # lin3

    img=imread(mypath+'K344.tif')

    pylab.imshow(img,zorder=0,extent=[260002,260002+4*12000,6665988-4*6000,6665988],alpha=alpha)

    pylab.hold(True)


    if show=='LA':

        pylab.xlim(xlim_LA)
        pylab.ylim(ylim_LA)

    elif show=='MY':

        pylab.xlim(xlim_MY)
        pylab.ylim(ylim_MY)

    else:

        pylab.xlim(xlim_all)
        pylab.ylim(ylim_all)


    # overlay nodes first

    if not(smallnodes):

        nodeSizes=get_nodesizes(nodedict,minsize=6.0,maxsize=10.0)
        nodeColors=get_nodecolors(nodedict,colormap='jet',minr=minr,maxr=maxr)

    else:

        nodeSizes={}
        nodeColors={}

        for node in net:

            nodeSizes[node]=nodesize
            nodeColors[node]='w'

    for node in net:        

        pylab.plot(coords[node][0],coords[node][1],'ko',color=nodeColors[node],markersize=nodeSizes[node],zorder=1000)


    # then links

    links_unsorted=list(net.edges)
    links=sorted(links_unsorted,key=lambda x:x[2])

    minlink=1.0
    maxlink=4.0

    if smallinks:

        minlink=0.2
        maxlink=1.0

    for index,link in enumerate(links):

        width=scaled(link[2],(minr,maxr),(minlink,maxlink))
        color=get_color(link[2],(minr,maxr))

        i=link[0]
        j=link[1]

        pylab.plot([coords[i][0],coords[j][0]],[coords[i][1],coords[j][1]],'k-',color=color,linewidth=width,zorder=index+1)

    pylab.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',
    top='on',   
    labelbottom='off',
    labeltop='off',
    labelright='off',
    labelleft='off') # labels along the bottom edge are off

    savestring='threshold_p%1.2f'%(pvalue)

    pylab.show()

    pylab.hold(False)

    pylab.savefig(mypath+savestring+filename+'.pdf')


def links(net,addone=True):

    '''Returns a list of the links of a network, with entries (nest1,nest2,relatedness)
       If addone=True, subtracts 1.0 to compensate for the +1.0 bias in SymmNets containing relatedness values'''

    if not addone:

          return list(net.edges)

    else:

          return [(x[0],x[1],x[2]-1.0) for x in list(net.edges)]

def joint_edges(net1,net2,addone=True):

    r1=[]
    r2=[]

    if addone:

        bias=1.0

    else:

        bias=0.0

    for edge in list(net1.edges):

        if (edge[0] in net2) and (edge[1] in net2):

            r1.append(edge[2]-bias)
            r2.append(net2[edge[0]][edge[1]]-bias)

    return r1,r2


# ==================== COMPUTING REFERENCE ENSEMBLE WITH ANTS SHUFFLED BETWEEN NESTS ================
#

def shuffled_rvalues_conserve(antdict,N,anttypes,reftypes):

    '''Shuffles ants between nests, computes new network, lists relatedness values, repeats N times.
       Keeps the number of ants of each type at each nest constant.'''

    rvalues=[]

    t00=time.time()

    for i in range(0,N):

        t0=time.time()


        dummy=shuffled_antdict_conserve(antdict)

        tempnet=antnet(dummy,anttypes=anttypes,reftypes=reftypes,colonyavg=False)

        rvalues.extend([edge[2] for edge in list(tempnet.edges)])

        delta_t=time.time()-t0

        print 'Network '+str(i)+' completed in time '+"{0:0.1f}".format(delta_t)+" s, estimated runtime "+"{0:0.1f}".format(((N-float(i))*delta_t)/60.0)+" min

def shuffled_rvalues(antdict,N,anttypes,reftypes):

    '''Shuffles ants between nests, computes new network, lists relatedness values, repeats N times.
       Does not keep the number of ants of each type at each nest constant.'''

    rvalues=[]

    t00=time.time()

    for i in xrange(0,N):

        t0=time.time()

        dummy=shuffled_antdict(antdict)

        tempnet=antnet(dummy,anttypes=anttypes,reftypes=reftypes,colonyavg=False)

        rvalues.extend([edge[2] for edge in list(tempnet.edges)])

        delta_t=time.time()-t0

        print 'Network '+str(i)+' completed in time '+"{0:0.1f}".format(delta_t)+" s, estimated runtime "+"{0:0.1f}".format(((N-float(i))*delta_t)/60.0)+" min"

    print "Done "+str(N)+" shuffles, runtime "+"{0:0.1f}".format((time.time()-t00)/60.0)+" minutes."

    file_io.picklesave(mypath+'tempdata.pic',rvalues)

    return rvalues

def shuffled_antdict(antdict):

    '''Returns an antdict where the sites of all ants have been randomly shuffled'''

    new_antdict={}

    sites=sorted([antdict[ant]['site'] for ant in antdict],key=lambda *args:np.random.rand())

    for ant,site in zip(antdict.keys(),sites):

        new_antdict[ant]['site']=site
        new_antdict['alleles']=antdict[ant]['alleles']
        new_antdict['loci']=antdict[ant]['loci']
        new_antdict['type']=antdict[ant]['type']
        new_antdict['colony']=antdict[ant]['colony']

    return new_antdict
        

def shuffled_antdict_conserve(antdict):

    # shuffles the sites (locations) of all ants
    # while keeping the total number of ants of each type
    # constant at each site.

    typelist=[]

    for ant in antdict:

        if not(antdict[ant]['type'] in typelist):

            typelist.append(antdict[ant]['type'])

    # list ants of each type *and* their sites

    listdict={}

    for anttype in typelist:

        listdict[anttype]=[[],[]] # construct a separate list of lists [ants],[sites] for each anttype

    for ant in antdict:

        listdict[antdict[ant]['type']][0].append(ant)
        listdict[antdict[ant]['type']][1].append(antdict[ant]['site'])

    for anttype in typelist:

        sitelist_temp=listdict[anttype][1]
        
        sitelist_new=sorted(sitelist_temp,key=lambda *args:np.random.rand()) # shuffle the site id list

        listdict[anttype][1]=sitelist_new

    new_antdict={}

    for anttype in typelist:

        for i in xrange(0,len(listdict[anttype][0])):

            ant=listdict[anttype][0][i]
            site=listdict[anttype][1][i]

            tempdict={}

            tempdict['alleles']=antdict[ant]['alleles']
            tempdict['loci']=antdict[ant]['loci']
            tempdict['type']=antdict[ant]['type']
            tempdict['colony']=antdict[ant]['colony']
            tempdict['site']=site

            new_antdict[ant]=tempdict

    return new_antdict

# ==================== within-colony relatednesses ==================================================

def r_within_colony_for_all(antdict,anttypes=["eworker","queen"],reftypes=["eworker","queen"],anttypes2=[],addone=False,colonyavg=False,px_missing=False,fullN=False,biascorrect=True,doublehomoz=True):

    ''' Calculates within-colony relatedness for all colonies in antdict
     if you wish to calculate queen-to-worker relatedness, use "queen" in anttypes and "eworker" in anttypes2
    
     colonyavg = (True/False) USE FALSE. if True, background frequencies calculated s.t. each colony has equal contribution (low-N colonies: ants get extra weight)
    
     px_missing = False -> one missing allele results in skipping locus, True -> one missing allele -> px=0.5
   
     fullN = True -> total allele counts include every loci even with missing data'''

    if addone:

        addthis=1.0 # because r-values have the offset in networks

    else:

        addthis=0.0

    # first list all sites

    if colonyavg:

        sitenumbers=list_sitenumbers(antdict,anttypes)

    else:

        sitenumbers={}

    sitelist=[]

    for ant in antdict:

        if not(antdict[ant]['site'] in sitelist) and (antdict[ant]['type'] in anttypes):

            sitelist.append(antdict[ant]['site'])

    # then create dictionary of r values

    sitedict={}

    for site in sitelist:

        r=r_withincolony(antdict,site,anttypes=anttypes,reftypes=reftypes,anttypes2=anttypes2,sitenumbers=sitenumbers,colonyavg=colonyavg,px_missing=px_missing,fullN=fullN,biascorrect=biascorrect,doublehomoz=doublehomoz)

        sitedict[site]=r+addthis

    return sitedict

# ==================== AUX ROUTINES FOR COMPUTING DIVERSE THINGS REQUIRED BY OTHER ROUTINES =========
#
#                       (typically you do not need to touch any of these)
#
# ===================================================================================================

def get_allele_counts(antdict,anttypes=['eworker','queen']):

    '''Yields dictionaries {site:allele_counts} {site:valid_counts}
       where allele_counts = {locus:N_alleles} where N_alleles={allele:count)
       and valid_counts = {locus:N_valid_alleles}
       valid = data not missing'''

    # gives dictionary where keys are loci
    # and values dictionaries where keys are sites
    # and values contain N of each allele

    # first get number of loci

    N_loci=len(antdict[antdict.keys()[0]]['alleles'])

    # get list of sites in data

    ant_sitedict=get_ant_sitedict(antdict,anttypes)

    # initialize some dictionaries
    # keys = sites, contain dicts where keys=loci

    allele_count={}
    total_valid_loci={}

    for site in ant_sitedict:

        locusdict={}
        validdict={}

        for locus in xrange(0,N_loci):

            N_alleles={}
            N_valids=0

            for ant in ant_sitedict[site]:

                thisant1=antdict[ant]['alleles'][locus][0]
                thisant2=antdict[ant]['alleles'][locus][1]

                if not(antdict[ant]['type']=='malepupa'):

                    if not(thisant1==-999):

                        if thisant1 in N_alleles:

                            N_alleles[thisant1]+=1

                        else:

                            N_alleles[thisant1]=1

                        N_valids+=1

                    if not(thisant2==-999):

                        if thisant2 in N_alleles:

                            N_alleles[thisant2]+=1

                        else:

                            N_alleles[thisant2]=1

                        N_valids+=1

                else: # if male pupa, add only one allele and only one valid allele

                    if not(thisant1==-999):

                        if thisant1 in N_alleles:

                            N_alleles[thisant1]+=1

                        else:

                            N_alleles[thisant1]=1

                        N_valids+=1

            locusdict[locus]=N_alleles
            validdict[locus]=N_valids

        allele_count[site]=locusdict
        total_valid_loci[site]=validdict

    return allele_count,total_valid_loci

def typestats(antdict,verbose=False,anttypes=['eworker','queen']):

    '''For checking statistics from antdict. Yields a count of different ant types
       for each site as dict {site:count}. If verbose = True, prints statistics.'''

    ant_sitedict=get_ant_sitedict(antdict,anttypes=anttypes)

    typelist=[]

    typecounts={}

    for ant in antdict:

        if not(antdict[ant]['type'] in typelist):

            typelist.append(antdict[ant]['type'])

    #print typelist

    for site in ant_sitedict:

        if verbose:

            print "Site ",site
            print "==========="

        typedict={}
        
        for anttype in typelist:
            typedict[anttype]=0

        for ant in ant_sitedict[site]:

            typedict[antdict[ant]['type']]+=1

        if verbose:

            for anttype in typedict:

                print anttype,typedict[anttype]

            print " "

        typecounts[site]=typedict

    return typecounts

def get_ant_sitedict(antdict,anttypes=['eworker','queen']):

    '''Yields dict {site:[list of ants]}'''

    sitelist=[]

    for ant in antdict:

        if not(antdict[ant]['site'] in sitelist):

            sitelist.append(antdict[ant]['site'])

    ant_sitedict={}

    for site in sitelist:

        ant_sitedict[site]=[]

    for ant in antdict:

        if antdict[ant]['type'] in anttypes:

            ant_sitedict[antdict[ant]['site']].append(ant)

    return ant_sitedict

def site_average_fraction(antdict,locus,allele,site,anttypes=['eworker','queen'],removeant=[],fullN=False):

    #counts the fraction of alleles "allele" at a specific locus
    #for all ants of site "site"; missing alleles are not
    #taken into account; average is over existing alleles only
    #if only over one ant type, set anttypes accordingly
    #
    # fullN = True -> frequencies are calculated for all loci s.t. even missing alleles contribute to N_alleles
    #

    N=0
    Nlocus=0


    for ant in antdict:

        if (antdict[ant]['site']==site) and (antdict[ant]['type'] in anttypes) and (not(ant in removeant)):

            allele1=antdict[ant]['alleles'][locus][0]
            allele2=antdict[ant]['alleles'][locus][1]
            
            if not(allele1==-999):

                N+=1 # increase checked alleles

                if allele1==allele:

                    Nlocus+=1 # increase found alleles

            elif fullN==True:

                N+=1

            if not(allele2==-999) and not(antdict[ant]['type']=='malepupa'): #haploid male pupa count as 1/2 individuals OK

                N+=1

                if allele2==allele:

                    Nlocus+=1

            elif fullN==True:

                N+=1

    py=float(Nlocus)/N

    return py


def r_withincolony(antdict,site,anttypes=["eworker","queen"],reftypes=["eworker","queen"],anttypes2=[],sitenumbers=[],biascorrect=True,colonyavg=False,px_missing=False,fullN=False,doublehomoz=True):

    '''Calculates relatedness within site. If you want to use all ant types for site-internal freqs, use anttypes2, otherwise anttypes2=anttypes.

       px_missing: if False, allele skipped if either allele1 or allele2 is missing
                   if True, if one known and one missing allele -> px=0.5'''

    if anttypes2==[]:

        anttypes2=anttypes

    antlist=[]

    for ant in antdict:

        if (antdict[ant]['site']==site) and (antdict[ant]['type'] in anttypes):

            antlist.append(ant)

    sum_self=[]
    sum_others=[]

    L=len(antdict[antlist[0]]['alleles'])

    for ant in antlist:

        for locus in range(0,L):

            focal_allele1=antdict[ant]['alleles'][locus][0]
            focal_allele2=antdict[ant]['alleles'][locus][1]

            if (not(focal_allele1==-999)) and (not(focal_allele2)==-999):

                if biascorrect:

                    removesite=site

                else:

                    removesite=[]

                wx=1.0

                pmx1=population_average_biascorrect(antdict,locus,focal_allele1,removesite=removesite,reftypes=reftypes)
                pmx2=population_average_biascorrect(antdict,locus,focal_allele2,removesite=removesite,reftypes=reftypes)

                if focal_allele1==focal_allele2:

                    if not(focal_allele1)==-999:

                        px=1.0

                        py=site_average_fraction(antdict,locus,focal_allele1,site,anttypes=anttypes2,removeant=[ant],fullN=fullN)

                        sum_self.append((px-pmx1))
                        sum_others.append((py-pmx1))

                        if doublehomoz and not(antdict[ant]['type']=='malepupa'):

                            sum_self.append((px-pmx1))
                            sum_others.append((py-pmx1))
                            

                else:

                        px=0.5

                        if not(focal_allele1)==-999:

                            py1=site_average_fraction(antdict,locus,focal_allele1,site,anttypes=anttypes2,removeant=[ant],fullN=fullN)

                            sum_self.append((px-pmx1))
                            sum_others.append((py1-pmx1))

                        if not(focal_allele2)==-999:

                            py2=site_average_fraction(antdict,locus,focal_allele2,site,anttypes=anttypes2,removeant=[ant],fullN=fullN)

                            sum_self.append((px-pmx2))
                            sum_others.append((py2-pmx2))

            elif px_missing==True:

                thisallele=-999

                if focal_allele1==-999 and (not(focal_allele2==-999)):

                    thisallele=focal_allele2

                elif focal_allele2==-999 and (not(focal_allele1==-999)):

                    thisallele=focal_allele1

                if not(thisallele==-999):

                    if biascorrect:

                        removesite=site

                    else:

                        removesite=[]

                    if colonyavg:

                        pmx=population_average_colonies(antdict,locus,thisallele,removesite=removesite,reftypes=reftypes,fullN=fullN)


                    else:

                        pmx=population_average_biascorrect(antdict,locus,thisallele,removesite=removesite,reftypes=reftypes)

                    px=0.5

                    py=site_average_fraction(antdict,locus,thisallele,site,anttypes=anttypes2,removeant=[ant],fullN=fullN)

                    sum_self.append(px-pmx)
                    sum_others.append(py-pmx)

    r=sum(sum_others)/sum(sum_self)

    return r

def r_populations(antdict,ant_sitedict,site_one,site_two,anttypes,reftypes,allele_counts,total_valid_loci,colonyavg=False,doublehomoz=True):

    ''' Computes relatedness with the Queller & Goodnight formula
      calculates relatedness between site_one and site_two
      only ants of types in list anttypes are accounted for when comparing allele frequencies
      for POPULATION frequencies, only ants of types in list reftypes are considered 
      sums up all contributions to nominator and denominator
      calculation over loci that have no missing data
      group averages over "site_two" ants that have no missing data for the focal loci
    
      for speeding up uses pre-computed allele_counts and total_valid_loci (see antnet)'''
  

    pself=[]
    pothers=[]

    antlist=[]

    # first list all ants in site_one that are of the required type ('eworker','queen', or both)
    
    for ant in antdict:
        if (antdict[ant]['site']==site_one) and (antdict[ant]['type'] in anttypes):
            antlist.append(ant)

    # loop through every listed ant

    for ant in antlist:

            for locus in xrange(0,len(antdict[ant]['alleles'])): # loop through loci

                focal_allele1=antdict[ant]['alleles'][locus][0]
                focal_allele2=antdict[ant]['alleles'][locus][1]

                if not(focal_allele1==-999) and not(focal_allele2==-999): # calculate contribution to r only if value not missing (missing=-999)

                    site=antdict[ant]['site']

                    # homozygous or heterozygous?

                    if focal_allele1==focal_allele2:

                        if not colonyavg:

                            # the typical and fast way of computing population average for x

                            pmx=popavg(allele_counts,total_valid_loci,locus,focal_allele1,removesite=[site,site_two])

                        else:

                            # does something strange, do not use

                            pmx=population_average_colonies(antdict,locus,focal_allele1,removesite=[site,site_two],reftypes=reftypes)
                        
                        px=1.0

                        pself.append(px-pmx)
                        
                        if doublehomoz and not(antdict[ant]['type']=='malepupa'): #homozygotes contribute 2 sum terms except for haploid male pupae 

                            pself.append(px-pmx)

                        pmy=pmx # bias correction removes both groups x and y, so the same for both always

                        py=siteavg(antdict,ant_sitedict,locus,focal_allele1,site_two,anttypes=anttypes)

                        pothers.append(py-pmy)
                        if doublehomoz and not(antdict[ant]['type']=='malepupa'):
                            pothers.append(py-pmy)

                    else:

                        px=0.5

                        if not colonyavg:

                            pmx1=popavg(allele_counts,total_valid_loci,locus,focal_allele1,removesite=[site,site_two])
                            pmx2=popavg(allele_counts,total_valid_loci,locus,focal_allele2,removesite=[site,site_two])

                        else:

                            pmx1=population_average_colonies(antdict,locus,focal_allele1,removesite=[site,site_two],reftypes=reftypes)
                            pmx2=population_average_colonies(antdict,locus,focal_allele2,removesite=[site,site_two],reftypes=reftypes)

                        pself.append(px-pmx1)
                        pself.append(px-pmx2)

                        pmy1=pmx1
                        pmy2=pmx2

                        py1=siteavg(antdict,ant_sitedict,locus,focal_allele1,site_two,anttypes=anttypes)
                        py2=siteavg(antdict,ant_sitedict,locus,focal_allele2,site_two,anttypes=anttypes)

                        pothers.append(py1-pmy1)
                        pothers.append(py2-pmy2)

    return sum(pothers)/sum(pself)

def popavg(allele_counts,valid_counts,locus,allele,removesite=[]):

    '''Computes population average frequency of allele at locus. Use precomputed allele_counts
       and valid_counts; to exclude site, include in removesite.'''

    N_found=0
    N_total=0

    for site in allele_counts:

        if not(site in removesite):

            if allele in allele_counts[site][locus]:

                N_found+=allele_counts[site][locus][allele]
                
            N_total+=valid_counts[site][locus]

    return float(N_found)/float(N_total)

def siteavg(antdict,ant_sitedict,locus,allele,site,anttypes=['eworker','queen'],removeant=[]):

    '''Computes frequency of allele at site at specific locus.'''

    N=0
    Nlocus=0

    for ant in ant_sitedict[site]:

        if (antdict[ant]['type'] in anttypes) and not(ant in removeant):

            allele1=antdict[ant]['alleles'][locus][0]
            allele2=antdict[ant]['alleles'][locus][1]
            
            if not(allele1==-999):

                N+=1 # increase checked alleles

                if allele1==allele:

                    Nlocus+=1 # increase found alleles

            if not(allele2==-999) and not(antdict[ant]['type']=='malepupa'): #haploid male pupa count as 1/2 individuals OK

                N+=1

                if allele2==allele:

                    Nlocus+=1

    if N==0:

        print 'zero N!',site,locus,allele,anttypes

    py=float(Nlocus)/N

    return py

def population_average_biascorrect(antdict,locus,allele,removesite=['none'],reftypes=['eworker','queen']):

    # calculates the population average for a specific locus
    # used for relatedness calculations
    # and a specific allele (fraction of alleles per allelic positions)
    # (2 positions per locus)
    # set removeant if one ant's contribution should be removed
    # set removesite to contain sites that are not included in the average
    # (typically for relatedness between colony_i and colony_j removesite=["colony_i","colony_j"] 
    # if pop average should be for one ant type only, one needs to set "reftypes" accordingly

    N_total=0
    N_allele=0

    for ant in antdict:

        site=antdict[ant]['site']

        thistype=antdict[ant]['type']

        if not(site in removesite) and (antdict[ant]['type'] in reftypes): # do not count sites in list removesite; count only ants of chosen reftype

            thisant1=antdict[ant]['alleles'][locus][0]
            thisant2=antdict[ant]['alleles'][locus][1]

            if not(antdict[ant]['type']=='malepupa'): # for diploids, check both alleles and increase counters

                if not(thisant1==-999):

                    if thisant1==allele: # if allele found, increase allele counter

                        N_allele+=1 

                    N_total+=1 # increase the counter for checked alleles

                if not(thisant2==-999):

                    if thisant2==allele:

                        N_allele+=1

                    N_total+=1
                    
            else: # for haploids (male pupae), only count as one instance of particular allele, both for frequency AND checked no of alleles

                if (thisant1==allele) and (not(thisant1==-999)):

                    N_allele+=1

                N_total+=1

    f_allele=float(N_allele)/float(N_total)

    return f_allele

def population_average_colonies(antdict,locus,allele,removesite=['none'],reftypes=['eworker','queen'],fullN=False):

    # calculates the population average for a specific locus
    # used for relatedness calculations
    # and a specific allele (fraction of alleles per allelic positions)
    # (2 positions per locus)
    # set removeant if one ant's contribution should be removed
    # set removesite to contain sites that are not included in the average
    # (typically for relatedness between colony_i and colony_j removesite=["colony_i","colony_j"] s.t.
    # only the other sites are used for the average)
    # if pop average should be for one ant type only, one needs to set "reftypes" accordingly

    # !!!! AVERAGES OVER SITES, EACH SITE HAS EQUAL WEIGHT INDEPENDENT OF THEIR NUMBER OF SAMPLED ANTS
    # modification of population_average_biascorrect
    #
    # fullN = True -> each locus counts to N regardless of whether allele missing or not


    N_total=0
    N_allele=0

    sitedict_alleles={}
    sitedict_totals={}

    # generate sitelist

    sitelist=[]

    for ant in antdict:

        if not(antdict[ant]['site']) in sitelist:

            if not(antdict[ant]['site']) in removesite:

                sitelist.append(antdict[ant]['site'])
            

    for site in sitelist:

        sitedict_alleles[site]=0 # total sought-for alleles per site
        sitedict_totals[site]=0  # total alleles per site                     

    for ant in antdict:

        site=antdict[ant]['site']

        thistype=antdict[ant]['type']

        if not(thistype=='malepupa'): # haploid male pupa only count as half individuals

            counts_as=1

        else:

            counts_as=0.5

        if (not(site in removesite)) and (thistype in reftypes): # do not count sites in list removesite; count only ants of chosen reftype

            thisant1=antdict[ant]['alleles'][locus][0]
            thisant2=antdict[ant]['alleles'][locus][1]

            if not(thisant1==-999):

                if thisant1==allele: # if allele found, increase allele counter

                    sitedict_alleles[site]+=counts_as # fixed 8.10.2013 - the two alleles of a haploid male pupa should count as 1

                sitedict_totals[site]+=counts_as  # increase the counter for checked alleles

            elif fullN:

                sitedict_totals[site]+=counts_as

            if not(thisant2==-999):

                if thisant2==allele:

                    sitedict_alleles[site]+=counts_as # fixed 8.10.2013 - the two alleles of a haploid male pupa should count as 1

                sitedict_totals[site]+=counts_as

            elif fullN:

                sitedict_totals[site]+=counts_as

    # now sum up site averages

    f_allele=0.0

    for site in sitelist:

        if not(sitedict_totals[site]==0):

            f_allele+=(1.0/float(len(sitelist)))*(float(sitedict_alleles[site])/float(sitedict_totals[site]))

        else:

            print 'Allele '+str(allele)+' not found at site '+str(site)+'\n'

    return f_allele

def net_subtract_one(net):

    '''Subtracts 1.0 from all edge values'''

    newnet=pynet.SymmNet()

    edges=list(net.edges)

    for edge in edges:

        newnet[edge[0]][edge[1]]=edge[2]-1.0

    for node in net:

        if not newnet.__contains__(node):

            newnet.addNode(node)

    return newnet

def threshold_by_pvalue(net,rvalues,pvalue,bonferroni=False,orig_addone=False,ref_addone=False):

    '''Thresholds network by p-value s.t. rvalues contains relatedness values from shuffled reference
       (with shuffled_ensemble_conserve())
       ensemble, and net is thresholded at the r where the p of finding it or larger in pvalues is pvalue'''

    if bonferroni:

        N=0
        for node in net:
            N+=1

        factor=0.5*float(N)*float((N-1))

        pvalue=pvalue/factor

    r_sorted=sorted(rvalues)

    L=len(r_sorted)

    rough_point=int(round((1.0-pvalue)*float(L)))

    threshold_value=r_sorted[rough_point]

    sigmas=(threshold_value-np.mean(r_sorted))/np.std(r_sorted)

    print "Threshold at sigma=",sigmas

    if orig_addone and not(ref_addone):

        threshold_value+=1.0

    print threshold_value,rough_point

    newnet=threshold_by_value(net,threshold_value,accept='>',keepIsolatedNodes=True)

    return newnet,threshold_value

def list_sitenumbers(antdict,anttypes):

    sitedict={}

    for ant in antdict:

        if antdict[ant]['type'] in anttypes:

            s=antdict[ant]['site']

            if not(s in sitedict):

                sitedict[s]=1

            else:

                sitedict[s]+=1

    return sitedict

def get_min_max_r(nets):

    '''returns the minimum, maximum, and mean relatedness values for a network or list of networks'''

    for net in nets:

        w=[x[2] for x in net.edges]

    return np.min(w),np.max(w),np.mean(w)

def get_nodesizes(property_dict,minsize=3.0,maxsize=12.0,minr=-999,maxr=-999):

    nodeSizes={}

    if minr==-999:

        maxr=max(property_dict.values())
        minr=min(property_dict.values())

    if maxr==minr:

        A=0

    else:

        A=(maxsize-minsize)/(maxr-minr)
        
    B=maxsize-A*maxr

    for node in property_dict:
        nodeSizes[node]=A*property_dict[node]+B

    return nodeSizes

def get_nodecolors(property_dict,colormap='jet',minr=-999,maxr=-999):

    myNodeColors=setColorMap(colormap)

    if minr==-999:

        maxr=max(property_dict.values())
        minr=min(property_dict.values())

    nodeColors={}

    for node in property_dict:

            nodeColors[node]=setColor(property_dict[node],(minr,maxr),myNodeColors)

    return nodeColors

def scaled(value, value_limits, final_limits,scaling_type='lin'):

    '''Scales value from range value_limits to range final_limits'''

    def lin_scaling(value, value_limits, final_limits):
        value_span = value_limits[1] - value_limits[0]
        final_span = final_limits[1] - final_limits[0]
        if final_span == 0:
            return final_limits[0]
        if value_span == 0:
            p = 0.5
        else:
            p = float(value - value_limits[0])/value_span
        return final_limits[0]+p*final_span

    if value <= value_limits[0]:
        return final_limits[0]
    if value >= value_limits[1]:
        return final_limits[1]

    if scaling_type == 'log' or scaling_type == 'logarithmic':
        return lin_scaling(np.log(value),
                           np.log(value_limits),
                           final_limits)
    else:
        return lin_scaling( value, value_limits, final_limits)

def get_color(value,value_limits,colormap='jet',scaling_type='lin'):

    myMap=pylab.get_cmap(colormap)

    return myMap(scaled(value,value_limits,final_limits=(0.0,1.0),scaling_type=scaling_type))

def normalizeValue(value,valueLimits):
    """Transforms a numerical value to the range (0,1).

    It is intended that the user should set valueLimits such that the
    true values fall between the limits. If this is not the case,
    values above given maxval or below given minval are truncated. The
    rest of the values are transformed linearly, such that the range
    (given minval, given maxval) becomes (0,1).
     
    normalizedValue= (true_val-given minval)/(given_maxval-given_minval)
    """ 
    if (valueLimits[0]-valueLimits[1]) == 0: 
        # If given minval and maxval are the same, all values will be
        # equal.
        normalizedValue=1
    elif value < valueLimits[0]:
        # If value is smaller than given minval
        normalizedValue=0
    elif value>valueLimits[1]:
        # If value is larger than given maxval
        normalizedValue=1
    else:
        normalizedValue=(value-valueLimits[0])/float(valueLimits[1] -
                                                     valueLimits[0])
    return normalizedValue 


def setColor(value,valueLimits,colorMap):
    """Converts a numerical value to a color.

    The value is scaled linearly to the range (0...1) using the
    function normalizeValue and the limits valueLimits. This scaled
    value is used to pick a color from the given colormap. The
    colormap should take in values in the range (0...1) and produce a
    three-tuple containing an RGB color, as in (r,g,b).
    """
    if valueLimits[0] < valueLimits[1]: 
        normalizedValue = normalizeValue(value,valueLimits)
        color = colorMap(normalizedValue)
    else:
        color=(0.5,0.5,0.5)  # gray if all values are equal
    return color

def setColorMap(colorMap):
    """Set a colormap for edges.

    Two options of our own ('orange' and 'primary') are available in
    addition to the 150 pylab readymade colormaps (which can be listed
    with help matplotlib.cm ).

    Usage:
        myMap = setColorMap('bone')
    """
    if hasattr(colorMap, '_segmentdata'):
        return colorMap

    known_colormaps = ('primary', 'orange', 'bluered')
    if colorMap in known_colormaps:
        if colorMap == 'primary':
            # yellow->blue->red 
            segmentdata={'red': ( (0,1,1),(0.5,0,0), (1,1,1)  ),
                         'green': ( (0,1,1), (0.5,0.5,0.5), (1,0,0) ),
                         'blue': ( (0,0,0), (0.5,1,1), (1,0,0) )}
        elif colorMap=='orange':
            # color map from white through yellow and orange to red 
            segmentdata = { 'red'  : ( (0.,.99,.99), 
                                       (0.2,.98,.98), 
                                       (0.4,.99,.99), 
                                       (0.6,.99,.99), 
                                       (0.8,.99,.99), 
                                       (1.0,.92,.92) ),
                            'green': ( (0,0.99,0.99), 
                                       (0.2,.89,.89),  
                                       (0.4,.80,.80), 
                                       (0.6,.50,.50), 
                                       (0.8,.33,.33), 
                                       (1.0,.10,.10) ),
                            'blue' : ( (0,.99,.99), 
                                       (0.2,.59,.59), 
                                       (0.4,.20,.20), 
                                       (0.6,0.0,0.0), 
                                       (0.8,0.0,0.0), 
                                       (1.0,.03,.03) )  }
        elif colorMap=='bluered':
            segmentdata={'red':  ( (0,0,0), 
                                   (0.17,0.25,0.25), 
                                   (0.33,0.7,0.7), 
                                   (0.5,.87,.87), 
                                   (0.67,.97,.97),  
                                   (0.83,.93,.93), 
                                   (1,.85,.85) ),
                         'green': ( (0,0,0), 
                                    (0.1667,0.53,0.53), 
                                    (0.3333,.8,.8), 
                                    (0.5,.9,.9), 
                                    (0.6667,.7,.7),
                                    (0.8333,.32,.32), 
                                    (1,.07,.07) ),
                         'blue': ( (0,.6,.6),  
                                   (0.1667,.8,.8),    
                                   (0.3333,1,1),    
                                   (0.5,.8,.8),    
                                   (0.6667,.33,.33),    
                                   (0.8333,.12,.12),
                                   (1,.05,.05) ) }
        myMap = matplotlib.colors.LinearSegmentedColormap(colorMap, segmentdata)
    else:
        try:
            myMap=pylab.get_cmap(colorMap)
        except AssertionError:
            comment = "Could not recognize given colorMap name '%s'" % colorMap
            raise AssertionError(comment)
    return myMap


def threshold_by_value(net,threshold,accept="<",keepIsolatedNodes=False):
    '''Generates a new network by thresholding the input network. 
       If using option keepIsolatedNodes=True, all nodes in the
       original network will be included in the thresholded network;
       otherwise only those nodes which have links will remain (this
       is the default). 
    
       Inputs: net = network, threshold = threshold value,
       accept = "foobar": accept weights foobar threshold (e.g accept = "<": accept weights < threshold)
       Returns a network.'''

    class NodeProperties(dict):
        def __init__(self):
            super(dict,self)
            self.__dict__={}
        def addProperty(self,propertyName):
            if not hasattr(self,propertyName):
                newValue={}
                self[propertyName]=newValue
                self.__setattr__(propertyName,newValue)
                
    def addNodeProperty(net,propertyName):
        if not hasattr(net,"nodeProperty"):
            net.nodeProperty=NodeProperties()
        net.nodeProperty.addProperty(propertyName)
        #if not hasattr(net.nodeProperty,propertyName):
        #    newValue={}
        #    #net.nodeProperty.__setattr__(propertyName,newValue)
        #    net.nodeProperty[propertyName]=newValue

    def copyNodeProperties(fromNet,toNet):
        if hasattr(fromNet,"nodeProperty"):
            for p in fromNet.nodeProperty:
                addNodeProperty(toNet,p)
                for node in toNet:
                    value=fromNet.nodeProperty[p][node]
                    toNet.nodeProperty[p][node]=value

    newnet=pynet.SymmNet()
    edges=list(net.edges)
    if accept == "<":
        for edge in edges:
            if (edge[2] < threshold):
                newnet[edge[0],edge[1]]=edge[2]
    elif accept == ">":
        for edge in edges:
            if (edge[2] > threshold):
                newnet[edge[0],edge[1]]=edge[2] 
    elif accept == ">=":
        for edge in edges:
            if (edge[2] >= threshold):
                newnet[edge[0],edge[1]]=edge[2] 
    elif accept == "<=":
        for edge in edges:
            if (edge[2] <= threshold):
                newnet[edge[0],edge[1]]=edge[2] 
    else:
        raise Exception("Parameter 'accept' must be either '<', '>', '<=' or '>='.")

    # Add isolated nodes to the network.
    if keepIsolatedNodes==True:
        for node in net:
            if not newnet.__contains__(node):
                newnet.addNode(node)
            
    copyNodeProperties(net,newnet)
                
    return newnet

def subdict(antdict,prefix='MY',from_nest=1,to_nest=10):

    allowed=[]

    for i in range(from_nest,to_nest+1):

        allowed.append(prefix+str(i))

    newdict={}

    for ant in antdict:

        if antdict[ant]['site'] in allowed:

            newdict[ant]=antdict[ant]

    return newdict

def get_min_max_r(nets,addone=True):

    '''returns the minimum, maximum, and mean relatedness values for a network or list of networks'''

    for net in nets:

        w=[x[2] for x in net.edges]

    if addone:

        return np.min(w)-1.0,np.max(w)-1.0,np.mean(w)-1.0

    else:

        return np.min(w),np.max(w),np.mean(w)

def between_nest_correlations(antdict,type1,type2,reftypes):

    net1=antnet(antdict,anttypes=type1,reftypes=reftypes)
    net2=antnet(antdict,anttypes=type2,reftypes=reftypes)

    r1,r2=joint_edges(net1,net2)

    print pearsonr(r1,r2)
    

def autothreshold(net):

    '''Yields the relatedness threshold for which the network splits into two components s.t. both are larger than
       one node'''

    edges=sorted(list(net.edges),key=lambda x:x[2])

    newnet=pynet.SymmNet()

    for edge in edges:

        newnet[edge[0]][edge[1]]=edge[2]

    stopnow=False

    i=0

    while not(stopnow):

        newnet[edges[i][0]][edges[i][1]]=0

        c=percolator.getComponents(newnet)

        if len(c)>1:

            sizes=[]

            for cx in c:

                sizes.append(len(cx))

            large_sizes=[s for s in sizes if s>1]

            if len(large_sizes)>1:

                stopnow=True
                
        i+=1

    return edges[i-1][2]

def cluster_MST(antnet,clusterA,clusterB):

    '''inputs ant net and its nests divided in clusters A and B (lists of nodes).
       Computes max spanning tree; computes how many of its links are in A-A,A-B,B-B.'''

    mst=transforms.mst_kruskal(antnet,maximum=True)

    internalA=0
    internalB=0
    between=0

    for link in list(mst.edges):

        if (link[0] in clusterA) and (link[1] in clusterA):

            internalA+=1

        elif (link[0] in clusterB) and (link[1] in clusterB):

            internalB+=1

        else:

            between+=1

    print "Internal links in A: ",internalA
    print "Internal links in B: ",internalB
    print "Between A and B: ",between

def split_rvalues(antnet,subtract_one=True):

    '''Input the full MY+LA antnet -> this function gives three lists of link weights, from the full MY and LA net:
       relatedness values in MY, LA and between: r_MY,r_LA, and r_between'''

    r_MY=[]
    r_LA=[]
    r_between=[]

    for edge in list(antnet.edges):

        if edge[0][0]=='M' and edge[1][0]=='M':

            r_MY.append(edge[2]-1.0)

        elif edge[0][0]=='L' and edge[1][0]=='L':

            r_LA.append(edge[2]-1.0)

        else:

            r_between.append(edge[2]-1.0)

    return r_MY,r_LA,r_between

def write_BAPS(antdict,filename,headerstring='',append=False,anttypes=['eworker','queen','lateworker','queenpupa','workerpupa']):

    '''Writes data that is originally in the antdict format used by these scripts as a file
       that is readable by BAPS. Set append=True to append to the end of an existing file'''

    separator=' ' # BAPS uses WS as separator

    if append:

        f=open(filename,"a")

    else:

        f=open(filename,"w")

        # first write header string
        
        f.write(headerstring+'\n')

        # then the names of all used loci

        loci=antdict[antdict.keys()[0]]['loci']

        for locus in loci:

            f.write(locus+'\n')

    # then each population in antdict

    # begin by listing all populations (sites) in antdict

    sites=[]

    for ant in antdict:

        if not(antdict[ant]['site'] in sites):

            sites.append(antdict[ant]['site'])

    # then go through each site

    for site in sites:

        f.write('Pop\n') 


        for ant in antdict:

            if antdict[ant]['site']==site and (antdict[ant]['type'] in anttypes):

                # write out each ant in BAPS format

                # first write out site

                f.write(site+','+separator)

                # then alleles

                for allele in antdict[ant]['alleles']:

                    if allele[0]==-999:

                        first='000'

                    else:

                        first=str(allele[0])

                    if allele[1]==-999:

                        second='000'

                    else:

                        second=str(allele[1])

                    if len(first)<3:

                        first='0'+first

                    if len(second)<3:

                        second='0'+second

                    f.write(first+second+separator)

                f.write('\n')

        
    f.close()

        





            

