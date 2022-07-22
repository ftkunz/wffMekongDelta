import pandas as pd
import uuid
import random
import pandas as pd
import numpy as np
import glob, os, json
import geopandas
# from pandas_geojson import read_geojson

import matplotlib.pyplot as plt
#
# rd = random.Random()
# #Change for different years:
# # 0 = 2021
# rd.seed(0)
# uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))
#
#
# # geodf = geopandas.read_file("boats_table_geojson_Zoom13_th0.2_Tile0_MekongDelta_2021.geojson")
#
# #Determine th offset - 0.2 was chosen!
# # lstn1 = []
# # lst0 = []
# # lst1 = []
# # for file in range(21):
# #     dfn1 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th-0.1_Tile" + str(file) + "_MekongDelta_2021.csv")
# #     df0 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0_Tile" + str(file) + "_MekongDelta_2021.csv")
# #     df1 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.1_Tile" + str(file) + "_MekongDelta_2021.csv")
# #
# #     sizen1 = max(dfn1.count())
# #     size0 =max(df0.count())
# #     size1 = max(df1.count())
# #     print(size1)
# #     mx = max(sizen1,size0,size1)
# #
# #     lstn1.append(sizen1/mx)
# #     lst0.append(size0 / mx)
# #     lst1.append(size1 / mx)
# #
# # print('th-01',np.mean(lstn1),'th0',np.mean(lst0),'th0.1',np.mean(lst1))
# #
# # lst01 = []
# # lst02 = []
# # lst03 = []
# # for file in range(21,29):
# #     df01 = pd.read_csv(
# #         "Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.1_Tile" + str(file) + "_MekongDelta_2021.csv")
# #     df02 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.2_Tile" + str(file) + "_MekongDelta_2021.csv")
# #     df03 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.3_Tile" + str(file) + "_MekongDelta_2021.csv")
# #
# #     size01 = max(df01.count())
# #     size02 = max(df02.count())
# #     size03 = max(df03.count())
# #     print(size01)
# #     mx = max(size01, size02, size03)
# #
# #     lst01.append(size01/mx)
# #     lst02.append(size02 / mx)
# #     lst03.append(size03 / mx)
# #
# # print('th0.1',np.mean(lst01),'th0.2',np.mean(lst02),'th0.3',np.mean(lst03))
# #
# # lst01 = []
# # lst02 = []
# # sum01 = 0
# # sum02 = 0
# # for file in range(30):
# #     df01 = pd.read_csv(
# #         "Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.1_Tile" + str(file) + "_MekongDelta_2021.csv")
# #     df02 = pd.read_csv("Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.2_Tile" + str(file) + "_MekongDelta_2021.csv")
# #
# #
# #     size01 = max(df01.count())
# #     size02 = max(df02.count())
# #     sum01 = sum01+size01
# #     sum02 = sum02 + size02
# #
# #     print(size01,size02)
# #     mx = max(size01, size02)
# #
# #     lst01.append(size01/mx)
# #     lst02.append(size02 / mx)
# #
# # print('th0.1',np.mean(lst01),sum01,'th0.2',np.mean(lst02),sum02) #0.2 gives most detected blobs
# #
# dfblob = []
# for file in range(271):
#     # geodf = geopandas.read_file("Mekong_delta_tables_geojson_2021/boats_table_geojson_Zoom13_th0.2_Tile"+str(file)+"_MekongDelta_2021.geojson")
#     geodf = pd.read_csv(
#         "Mekong_delta_tables_csv/boats_table_csv_Zoom13_th0.2_Tile" + str(file) + "_MekongDelta_2021.csv")
#     geodf['Tile'] = file
#     # geodf = geodf.assign(blobID = lambda x: uuid.uuid4())
#     dfblob.append(geodf)
# dfblob = pd.concat(dfblob)
# dfblob['blobID'] = dfblob.apply(lambda x: uuid.uuid4(), axis=1)
# dfblob.to_csv('Blobs2021.csv')
# #
# df = pd.read_csv('Blobs2021.csv')
# df.to_pickle('Blobs2021df')

df = pd.read_pickle('Blobs2021df')
df = df[df['height']<80]
df = df[df['width']<15]
df = df[df['distance2shore']>100]
# df = df[df['area']>1]
# df = df[df['area']<5000]
print(df['blobID'])
print(df['date'])
print(df['date'].value_counts())
print(df['Tile'].value_counts())
NoB = pd.DataFrame(df['date'].value_counts())
print(NoB)
NoB.to_csv('NumberofBlobs2021.csv')
# df.hist()
# df.hist('distance2shore',bins=100)
# df.hist('area',bins=100)
# df.hist('width',bins=100)
# df.hist('height',bins=100)
# plt.show()
TileVMD = [21,23,24,30,31,32,33,36,37,38,39,40,41,48,49,50,51,52,53,54,55,63,64,65,66,68,69,70,71,74,75,76,77,78,79,80,
           81,82,84,85,86,87,88,91,92,93,94,95,100,101,102,109,110,111,112,115,116,117,118,119,122,123,124,126,127,128,130,131,132,
           134,135,136,137,138,140,141,142,144,145,146,147,149,150,152,153,154,158,159,160,161,162,173,174,175,176,177,179,180,181,
           182,183]
for i in np.arange(185,271):
    TileVMD.append(i)
print(TileVMD)
df = df[df['Tile'].isin(TileVMD)]
area = df.groupby(['date']).agg({'area': 'sum'}).reset_index()
area['days'] = pd.to_datetime(area['date']).diff(-1)
print(area['days'])
area['volume']=area['area']*7
area['msand']=area['volume']*0.4
area['Msand']=area['msand']*-(area['days'].dt.days)
print(area['Msand'].sum(min_count=1))
print(area)

# print(max(df.count()))
#
# # df.hist('area',bins=100)
# # df.hist('width',bins=100)
#
# df = df[df['height']<200]
# df = df[df['distance2shore']>60]

# df.hist('height',bins=100)
# df.hist('area',bins=100)
# df.hist('width',bins=100)
# print(max(df.count()))
# # plt.show()
# print('distance2shore',np.mean(df['distance2shore']),max(df['distance2shore']),min(df['distance2shore']))
# print('area',np.mean(df['area']),max(df['area']),min(df['area']))
# print('width',np.mean(df['width']),max(df['width']),min(df['width']))
# print('height',np.mean(df['height']),max(df['height']),min(df['height']))