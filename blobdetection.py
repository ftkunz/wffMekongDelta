"""
This algorithm takes uses Sentinel-2 images to detect the boats in the Mekong Delta using the JRC Global Surface Water dataset.

The input to the detectblob function is the image, the bounds, the shore mask, the water mask and the offset to the found Otso threshold value.

The output is a FeatureCollection of the detected blobs with the following properties:
    'image_id': idimg,
    'threshold': ee.Number(th).add(extrath),
    'timestamp': timestamp,
    'date': date,
    'geometry': f.geometry(),
    'bounds': bounds,
    'width': width,
    'height': height,
    'centroid': centroid,
    'cenlon': ee.Number(ee.Geometry(centroid).coordinates().get(0)),
    'cenlat': ee.Number(ee.Geometry(centroid).coordinates().get(1)),
    'distance2shore': distance,
    'max_swir2': maxswir2,
    'mean_swir2': mswir2,
    'min_swir2': minswir2,
    'max_swir1': maxswir1,
    'mean_swir1': mswir1,
    'min_swir1': minswir1,
    'max_nir': maxnir,
    'mean_nir': mnir,
    'min_nir': minnir,
    'max_red': maxred,
    'mean_red': mred,
    'min_red': minred,
    'max_green': maxgreen,
    'mean_green': mgreen,
    'min_green': mingreen,
    'max_blue': maxblue,
    'mean_blue': mblue,
    'min_blue': minblue
"""


def detectblob(image, bounds, shore, water, extrath):
    # ee.Image,ee.Geometry,ee.Image(Mask),ee.Image(Mask),ee.Number
    date = image.date().format('YYYY-MM-dd')
    idimg = image.get('system:id')
    timestamp = image.get("system:time_start")

    cannyThreshold = 0.035
    cannySigma = 1
    minValue = -0.1
    scale = 3

    gshore = ee.Image.constant(1).clip(bounds).addBands(shore)
    gshore = gshore.updateMask(gshore.mask().multiply(shore))

    geoshore = gshore.reduceToVectors(
        reducer=ee.Reducer.allNonZero(),
        geometry=bounds,
        scale=5,
    )

    ndwi = image.normalizedDifference(['green', 'nir'])

    # #Uncomment to sharpen the image, which increases the blob detection but slows down the algorithm.
    # fat = ee.Kernel.gaussian(
    #     radius=3,
    #     sigma=3,
    #     magnitude=-1,
    #     units='meters')

    # skinny = ee.Kernel.gaussian(
    #     radius=3,
    #     sigma=0.2,
    #     units='meters')

    # dog = fat.add(skinny)
    # ndwi = ndwi.add(ndwi.convolve(dog));

    th = computeThresholdUsingOtsu(ndwi.updateMask(water),
                                   scale,
                                   bounds,
                                   cannyThreshold,
                                   cannySigma,
                                   minValue,
                                   False,
                                   False,
                                   False,
                                   False)

    waterMask = ndwi.gt(ee.Number(th).add(extrath))

    # #Uncomment to remove noise but decrease the speed
    # mask = waterMask.focalMin(30, 'circle', 'meters').focalMax(30, 'circle', 'meters')
    # waterMask = waterMask.multiply(mask)


    blobs = waterMask.Not().selfMask()
    blobs = blobs.updateMask(blobs.mask().multiply(water))

    blobsVector = blobs.reduceToVectors(
        geometry=bounds,
        scale=5,
        # eightConnected = True
    )

    blobsVector = blobsVector.map(lambda f: f.set('area', f.geometry().area(1)))

    minArea = 0
    maxArea = 8000
    # blobsVector = blobsVector.filter(ee.Filter.greaterThan({leftField:'area',rightField:'minArea'}))
    blobsVector = blobsVector.filter(ee.Filter.lt('area', maxArea))

    dist = blobsVector.distance(200).updateMask(blobs.mask())
    blobsVector = blobsVector.map(lambda f: f.set('halfWidth', ee.Number(dist.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=f.geometry(),
        scale=5,
    ).values().get(0))))
    blobsVector = blobsVector.filter(ee.Filter.gt('halfWidth', 0))


    #function to define the properties to be stored for all detected blobs
    def properties(f):
        bounds = f.geometry().buffer(5).bounds()
        centroid = f.geometry().centroid(1)

        width = ee.Number(f.get('halfWidth')).multiply(2)
        height = ee.Number(f.get('area')).divide(ee.Number(width))
        distance = ee.Geometry(centroid).distance(right=ee.Geometry(geoshore.geometry()), maxError=1)

        mswir2 = ee.Number(image.select(['swir2']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=20
        ).get('swir2'))

        mswir1 = ee.Number(image.select(['swir1']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=20
        ).get('swir1'))

        mnir = ee.Number(image.select(['nir']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=10
        ).get('nir'))

        mred = ee.Number(image.select(['red']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=10
        ).get('red'))

        mgreen = ee.Number(image.select(['green']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=10
        ).get('green'))

        mblue = ee.Number(image.select(['blue']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=10
        ).get('blue'))
        maxswir2 = ee.Number(image.select(['swir2']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=20
        ).get('swir2'))

        maxswir1 = ee.Number(image.select(['swir1']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=20
        ).get('swir1'))

        maxnir = ee.Number(image.select(['nir']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=10
        ).get('nir'))

        maxred = ee.Number(image.select(['red']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=10
        ).get('red'))

        maxgreen = ee.Number(image.select(['green']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=10
        ).get('green'))

        maxblue = ee.Number(image.select(['blue']).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=f.geometry(),
            scale=10
        ).get('blue'))
        minswir2 = ee.Number(image.select(['swir2']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=20
        ).get('swir2'))

        minswir1 = ee.Number(image.select(['swir1']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=20
        ).get('swir1'))

        minnir = ee.Number(image.select(['nir']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=10
        ).get('nir'))

        minred = ee.Number(image.select(['red']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=10
        ).get('red'))

        mingreen = ee.Number(image.select(['green']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=10
        ).get('green'))

        minblue = ee.Number(image.select(['blue']).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=f.geometry(),
            scale=10
        ).get('blue'))
        return f.set({
            'image_id': idimg,
            'threshold': ee.Number(th).add(extrath),
            'timestamp': timestamp,
            'date': date,
            'geometry': f.geometry(),
            'bounds': bounds,
            'width': width,
            'height': height,
            'centroid': centroid,
            'cenlon': ee.Number(ee.Geometry(centroid).coordinates().get(0)),
            'cenlat': ee.Number(ee.Geometry(centroid).coordinates().get(1)),
            'distance2shore': distance,
            'max_swir2': maxswir2,
            'mean_swir2': mswir2,
            'min_swir2': minswir2,
            'max_swir1': maxswir1,
            'mean_swir1': mswir1,
            'min_swir1': minswir1,
            'max_nir': maxnir,
            'mean_nir': mnir,
            'min_nir': minnir,
            'max_red': maxred,
            'mean_red': mred,
            'min_red': minred,
            'max_green': maxgreen,
            'mean_green': mgreen,
            'min_green': mingreen,
            'max_blue': maxblue,
            'mean_blue': mblue,
            'min_blue': minblue
        })

    blobsVector = blobsVector.map(properties)

    return blobsVector


"""
Algorithms by Gennadii Donchyts
"""
#Gena's thresholding module:
def otsu(histogram):
    histogram = ee.Dictionary(histogram)

    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    # Compute between sum of squares, where each mean partitions the data.

    def func_sgl(i):
        aCounts = counts.slice(0, 0, i)
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
        aMeans = means.slice(0, 0, i)
        aMean = aMeans.multiply(aCounts) \
                .reduce(ee.Reducer.sum(), [0]).get([0]) \
                .divide(aCount)
        bCount = total.subtract(aCount)
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(
                bCount.multiply(bMean.subtract(mean).pow(2)))

    bss = indices.map(func_sgl)

    # Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])
#**
 # Compute a threshold using Otsu method (bimodal)
 #
def computeThresholdUsingOtsu(image, scale, bounds, cannyThreshold, cannySigma, minValue, debug, minEdgeLength, minEdgeGradient, minEdgeValue):
    # clip image edges
    mask = image.mask().gt(0).clip(bounds).focal_min(ee.Number(scale).multiply(3), 'circle', 'meters')

    # detect sharp changes
    edge = ee.Algorithms.CannyEdgeDetector(image, cannyThreshold, cannySigma)
    edge = edge.multiply(mask)

    if(minEdgeLength):
        connected = edge.mask(edge).lt(cannyThreshold).connectedPixelCount(200, True)

        edgeLong = connected.gte(minEdgeLength)

        edge = edgeLong


    # buffer around NDWI edges
    edgeBuffer = edge.focal_max(ee.Number(scale), 'square', 'meters')

    if(minEdgeValue):
      edgeMin = image.reduceNeighborhood(ee.Reducer.min(), ee.Kernel.circle(ee.Number(scale), 'meters'))

      edgeBuffer = edgeBuffer.updateMask(edgeMin.gt(minEdgeValue))


    if(minEdgeGradient):
      edgeGradient = image.gradient().abs().reduce(ee.Reducer.max()).updateMask(edgeBuffer.mask())

      edgeGradientTh = ee.Number(edgeGradient.reduceRegion(ee.Reducer.percentile([minEdgeGradient]), bounds, scale).values().get(0))

      edgeBuffer = edgeBuffer.updateMask(edgeGradient.gt(edgeGradientTh))

    edge = edge.updateMask(edgeBuffer)
    edgeBuffer = edge.focal_max(ee.Number(scale).multiply(1), 'square', 'meters')
    imageEdge = image.mask(edgeBuffer)


    # compute threshold using Otsu thresholding
    buckets = 100
    hist = ee.Dictionary(ee.Dictionary(imageEdge.reduceRegion(ee.Reducer.histogram(buckets), bounds, scale)).values().get(0))

    threshold = ee.Algorithms.If(hist.contains('bucketMeans'), otsu(hist), minValue)
    threshold = ee.Number(threshold)

    if minValue != 'undefined':
        return threshold.max(minValue)
    else:
        return threshold

# Gena cloudfilter without options
def addQualityScore(images, g):
    scorePercentile = 75
    scale = 500
    mask = None
    qualityBand = 'green'

    def quality(i):
        score = i.select(qualityBand)
        score = score.reduceRegion(ee.Reducer.percentile([scorePercentile]), g, scale).values().get(0)
        return i.set({'quality_score': score})

    return images.map(quality)

def getMostlyCleanImages(images, g):
    g = ee.Geometry(g)

    scale = 500
    p = 85

    # http:#www.earthenv.Org/cloud
    modisClouds = ee.Image('users/gena/MODCF_meanannual')

    cloudFrequency = modisClouds.divide(10000).reduceRegion(
        ee.Reducer.percentile([p]),
        g.buffer(10000, scale * 10), scale * 10).values().get(0)

    # print('Cloud frequency (over AOI):', cloudFrequency)

    # decrease cloudFrequency, include some more partially-cloudy images then clip based on a quality metric
    # also assume inter-annual variability of the cloud cover
    cloudFrequency = ee.Number(cloudFrequency).subtract(0.15).max(0.0)

    images = images.filterBounds(g)

    size = images.size()

    images = addQualityScore(images, g).filter(ee.Filter.gt('quality_score', 0))

    #
    scoreMin = 0.01
    scoreMax = images.reduceColumns(ee.Reducer.percentile([ee.Number(1).subtract(cloudFrequency).multiply(100)]),
                                    ['score']).values().get(0)

    # filter by quality score
    # images = images
    #   .filter(ee.Filter.And(ee.Filter.gte('score', scoreMin), ee.Filter.lte('score', scoreMax)))
    #

    # clip collection
    images = images.sort('quality_score').limit(images.size().multiply(ee.Number(1).subtract(cloudFrequency)).toInt())

    # remove too dark images
    # images = images.sort('quality_score', False)
    # .limit(images.size().multiply(0.99).toInt())

    # print('size, filtered: ', images.size())

    return images


"""
How to use the function is shown below:
"""
import ee
import numpy as np

ee.Initialize()

#Define year
year = [2016,2017,2018,2019,2020,2021,2022]
y=5
#or
#for y in range(6):

#Define bounds
tiles = ee.FeatureCollection('zoom13tilesMD.csv')
listtiles = tiles.toList(tiles.size())
t=0
#or
#for t in range(listtiles.size().getInfo()):
bounds = ee.Feature(listtiles.get(t)).geometry()
# print(bounds.getInfo())

#Define shore & water mask
jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
jrcwater = jrc.select('occurrence').unmask(0).resample('bilinear').divide(100)
jrcwater = jrcwater.updateMask(jrcwater.gt(0.5))

shore = ee.Algorithms.CannyEdgeDetector(jrcwater, 0.1, 0.1).selfMask()
shore = shore.updateMask(shore.gt(1)).unmask()

#Get images
images = getMostlyCleanImages(ee.ImageCollection('COPERNICUS/S2')
            .filterBounds(bounds)
            .filterDate(str(year[y])+'-01-01', str(year[y]+1)+'-01-01')
            # .filterDate('2021-01-01', '2022-01-01')
            # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))
            .select(['B4', 'B3', 'B2', 'B8', 'B11','B12'], ['red', 'green', 'blue', 'nir', 'swir1','swir2'])
            .map(lambda i: i.resample('bicubic').divide(10000).set({ 'system:time_start': i.get('system:time_start') }).set({ 'system:id': i.get('system:id') })
            ),bounds)

images_list = images.toList(images.size())

#use detectblob function to create FeatureCollection of all detected blobs
extrath = 0
for i in range(images.size().getInfo()):
    image = ee.Image(images_list.get(i))
    blobs = detectblob(image,bounds,shore,jrcwater,extrath)
    if (i == 0):
        blobtable = ee.FeatureCollection(blobs)
    else:
        blobtable = blobtable.merge(blobs)