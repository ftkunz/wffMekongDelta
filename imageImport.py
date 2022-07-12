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

items = df['blobID']

ID = items[0]

df = df.loc[df['blobID']==ID]
print(str(df['image_id'].values[0]))
imgid = str(df['image_id'].values[0])
centroid = df['centroid'].values[0]
# print(centroid)
print(json.loads(centroid)['coordinates'])

centerpoint = ee.Geometry.Point(json.loads(centroid)['coordinates'])
region = centerpoint.buffer(100).bounds()

def stretchImage(image, scale ,bounds):
  percentiles = [1, 99]
  bandNames = image.bandNames()
  scale =2*scale
  bounds = bounds

  minMax = image.reduceRegion(
    reducer= ee.Reducer.percentile(percentiles),
    geometry= bounds,
    scale= scale
    )


  def func_sfu(bandName):
      bandName = ee.String(bandName)
      min = ee.Number(minMax.get(bandName.cat('_p').cat(ee.Number(percentiles[0]).format())))
      max = ee.Number(minMax.get(bandName.cat('_p').cat(ee.Number(percentiles[1]).format())))

      return ee.Number(image.select(bandName)).subtract(min).divide(max.subtract(min))

  bands = bandNames.map(func_sfu)

  return ee.ImageCollection(bands).toBands().rename(bandNames)

# print(bounds.getInfo())
image = (stretchImage(ee.Image(imgid)
        .clip(region).select(['B4','B3','B2'])
                 .resample('bicubic').divide(10000),3,region)
                    .visualize({min:0,max:1}))

url = image.getThumbURL({
    'region': region,
    'dimensions': '256x256',
    'format': 'png'})

r = requests.get(url, stream=True)
if r.status_code != 200:
    r.raise_for_status()

filename = 'BlobImg/'+str(ID)+'.png'
with open(filename,'wb') as out_file:
    shutil.copyfileobj(r.raw, out_file)
print("Done")


#
# @retry(tries=10,delay=1,backoff=2)
# def getResult(index,blobID):
#     df