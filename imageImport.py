import ee
import logging
import multiprocessing
import requests
import shutil
from retry import retry
import pandas as pd
import json

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
df = pd.read_pickle('Blobs2021df')


def stretchImage(image, scale ,bounds):
  percentiles = [1, 99]
  bandNames = image.bandNames()
  scale =2*scale
  bounds = bounds
  imageMask = image.select(0).mask()

  minMax = image.updateMask(imageMask).reduceRegion(
    reducer= ee.Reducer.percentile(percentiles),
    geometry= bounds,
    scale= scale
    )


  def func_sfu(bandName):
      bandName = ee.String(bandName)
      min = ee.Number(minMax.get(bandName.cat('_p').cat(ee.Number(percentiles[0]).format())))
      max = ee.Number(minMax.get(bandName.cat('_p').cat(ee.Number(percentiles[1]).format())))

      return image.select(bandName).subtract(min).divide(max.subtract(min))

  bands = bandNames.map(func_sfu)

  return ee.ImageCollection(bands).toBands().rename(bandNames)


@retry(tries=10,delay=1,backoff=2)
def getResult(index,blobID):

    df = df.loc[df['blobID'] == blobID]
    imgid = str(df['image_id'].values[0])
    centroid = df['centroid'].values[0]

    centerpoint = ee.Geometry.Point(json.loads(centroid)['coordinates'])
    region = centerpoint.buffer(100).bounds()

    image = stretchImage(ee.Image(imgid)
                        .clip(region)
                         .select(['B4', 'B3', 'B2'])
                            # .select(['B12','B8','B3'])
                        .resample('bicubic').divide(10000), 3, region).visualize(min=0, max=1)

    url = image.getThumbURL({
        'region': region,
        'dimensions': '256x256',
        'format': 'png'})

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    filename = 'BlobpngRGB/' + str(ID) + '.png'
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done")

if __name__ == '__main__':
  logging.basicConfig()
  items = df['blobID']

  pool = multiprocessing.Pool(25)
  pool.starmap(getResult, enumerate(items))

  pool.close()