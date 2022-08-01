import pandas as pd
import uuid
import random
import pandas as pd
import numpy as np
import glob, os, json
import geopandas
import matplotlib.pyplot as plt
# from pandas_geojson import read_geojson

import matplotlib.pyplot as plt
#
rd = random.Random()
#Change for different years:
# 0 = 2021, 1 = 2018, 2 = 2019, 3 = 2020, 4 = 2022
years = [2021,2018,2019,2020,2022] #Sorry weird order but otherwise the unique id for 2021 is not correct after I downloaded a lot of images already
year = 4
rd.seed(year)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))


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
# # #
# dfblob = []
# for file in range(271):
#     # geodf = geopandas.read_file("Mekong_delta_tables_geojson_2021/boats_table_geojson_Zoom13_th0.2_Tile"+str(file)+"_MekongDelta_2021.geojson")
#     geodf = pd.read_csv(
#         "Mekong_delta_tables_csv_"+str(years[year])+"/boats_table_csv_Zoom13_th0.2_Tile" + str(file) + "_MekongDelta_"+str(years[year])+".csv")
#     geodf['Tile'] = file
#     # geodf = geodf.assign(blobID = lambda x: uuid.uuid4())
#     dfblob.append(geodf)
# dfblob = pd.concat(dfblob)
# dfblob['blobID'] = dfblob.apply(lambda x: uuid.uuid4(), axis=1)
# dfblob.to_csv('Blobs'+str(years[year])+'.csv')
# #
# df = pd.read_csv('Blobs'+str(years[year])+'.csv')
# df.to_pickle('Blobs'+str(years[year])+'df')

from datetime import datetime
import pytz

verify = True
if verify:
    df = pd.read_pickle('Blobs'+str(years[year])+'df')
    df.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    # print(datetime.fromtimestamp(df['timestamp']))
    timestamp = 1545730073
    dt_object = datetime.fromtimestamp(timestamp)

    print("dt_object =", dt_object)
    print("type(dt_object) =", type(dt_object))
    print(df['timestamp'])
    df['DateTime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(int(x)/1000))
    print(df['DateTime'].drop_duplicates())
    file1 = open("DateTime", "a")
    file1.close()
    # timestamps = list(df['timestamp'])
    # dtlist = []
    # for i in range(len(timestamps)):
    #     dtlist.append(datetime.fromtimestamp(int(timestamps[i])))
    # print(dtlist)

    TileVMD = [21,23,24,30,31,32,33,36,37,38,39,40,41,48,49,50,51,52,53,54,55,63,64,65,66,68,69,70,71,74,75,76,77,78,79,80,
           81,82,84,85,86,87,88,91,92,93,94,95,100,101,102,109,110,111,112,115,116,117,118,119,122,123,124,126,127,128,130,131,132,
           134,135,136,137,138,140,141,142,144,145,146,147,149,150,152,153,154,158,159,160,161,162,173,174,175,176,177,179,180,181,
           182,183]
    for i in np.arange(185,271):
        TileVMD.append(i)
    print(TileVMD)
    SaDec2VinhLong = [134,135,136,140,144,149,152]
    KampongCham = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,26,27,28,29,34,35,36,37,38,39,40,
               42,43,44,45,46,47,48,49,50,56,57,58,59,60,61,62,67]

    # df = df[df['Tile'].isin(SaDec2VinhLong)]
    # df = df[df['Tile'].isin(KampongCham)]
    df = df[df['Tile'].isin(TileVMD)]

    dfold = df
    # try to fit to Jordan 2019
    df = df[df['height']<80]
    df = df[df['height']>30]
    df = df[df['width']<15]
    df = df[df['width']>6]
    df = df[df['height']>3.5*df['width']]
    df = df[df['distance2shore']>100]

    # df.to_csv('filteredBlobs'+str(years[year])+'')
    # df = df[df['area']>1]
    # df = df[df['area']<5000]
    print(df['blobID'])
    print(df['date'])
    print(df['date'].value_counts())
    print(df['Tile'].value_counts())
    NoB = pd.DataFrame(df['date'].value_counts())
    print(NoB)
    NoB.to_csv('NumberofBlobs'+str(years[year])+'.csv')
    # df.hist()
    # df.hist('distance2shore',bins=100)
    # df.hist('area',bins=100)
    # df.hist('width',bins=100)
    # df.hist('height',bins=100)
    # plt.show()


    area = df.groupby(['date']).agg({'area': 'sum'}).reset_index()
    area['days'] = pd.to_datetime(area['date']).diff(-1)
    print(area['days'])
    area['volume']=area['area']*7*0.39
    area['msand']=area['volume']
    area['Msand']=area['msand']*-(area['days'].dt.days)
    corr = 0.57
    print('method 1',corr*area['Msand'].sum(min_count=1))
    print(area)

    dfbc = dfold[dfold['distance2shore']>50]
    dfbc = dfbc[dfbc['height']>30]
    dfbc = dfbc[dfbc['height']<44]
    dfbc = dfbc[dfbc['area']>352]
    dfbc = dfbc[dfbc['area']<548]

    Abc = dfbc.groupby(['date']).agg({'area': 'sum'}).reset_index()
    getCount = dfbc.groupby(['date']).size().reset_index(name='count')
    print(getCount)
    Abc['days'] = pd.to_datetime(Abc['date']).diff(-1)

    Abc['count'] = getCount['count']

    print(len(Abc['count']),Abc['count'])
    Abc['volume300'] = -(Abc['days'].dt.days)*Abc['count']*300
    Abc['volume600'] = -(Abc['days'].dt.days)*Abc['count']*600
    Abc['volume']=Abc['area']*7*0.39*0.3
    Abc['msand']=Abc['volume']
    Abc['Msand']=Abc['msand']*-(Abc['days'].dt.days)
    Mbc = Abc['Msand'].sum(min_count=1)
    print('bc',Abc['count'])


    print('method 3',Abc['volume300'].sum(min_count=1),Abc['volume600'].sum(min_count=1),Abc.head())

    Abc = Abc.loc[(Abc['date'] >= '2022-03-01')
                      & (Abc['date'] < '2022-05-01')]
    print('april',Abc['date'],Abc['count'])

    dfbt = dfold[dfold['distance2shore']>50]
    dfbt = dfbt[dfbt['height']>44]
    dfbt = dfbt[dfbt['height']<50]
    dfbt = dfbt[dfbt['area']>505]
    dfbt = dfbt[dfbt['area']<611]

    Abt = dfbt.groupby(['date']).agg({'area': 'sum'}).reset_index()
    Abt['days'] = pd.to_datetime(Abt['date']).diff(-1)
    Abt['volume']=Abt['area']*7*0.39
    Abt['msand']=Abt['volume']
    Abt['Msand']=Abt['msand']*-(Abt['days'].dt.days)
    Mbt = Abt['Msand'].sum(min_count=1)

    dfbb = dfold[dfold['distance2shore']>50]
    dfbb = dfbb[dfbb['height']>39]
    dfbb = dfbb[dfbb['height']<55]
    dfbb = dfbb[dfbb['area']>288]
    dfbb = dfbb[dfbb['area']<464]

    Abb = dfbb.groupby(['date']).agg({'area': 'sum'}).reset_index()
    Abb['days'] = pd.to_datetime(Abb['date']).diff(-1)
    Abb['volume']=Abb['area']*7*0.39
    Abb['msand']=Abb['volume']
    Abb['Msand']=Abb['msand']*-(Abb['days'].dt.days)
    Mbb = Abb['Msand'].sum(min_count=1)

    print((Mbc+Mbt+Mbb))

    # print((Mbc+Mb)*1600)

    dfbc['area'] = 0.33*dfbc['area']
    DF = pd.concat([dfbb,dfbt,dfbc], ignore_index=True)
    DF.drop_duplicates(subset= ['timestamp','cenlon','cenlat'],ignore_index=True)
    # print(DF)

    A = DF.groupby(['date']).agg({'area': 'sum'}).reset_index()
    A['days'] = pd.to_datetime(A['date']).diff(-1)
    A['volume']=A['area']*7*0.39
    A['msand']=A['volume']
    A['Msand']=A['msand']*-(A['days'].dt.days)
    M = A['Msand'].sum(min_count=1)

    print('method 2',M)
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

df2018 = pd.read_pickle('Blobs2018df')
df2018.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
# June2018 =df2018.loc[(df2018['date'] >= '2018-01-01')
#                      & (df2018['date'] < '2018-06-01')]
df2019 = pd.read_pickle('Blobs2019df')
df2019.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
# June2019 =df2019.loc[(df2019['date'] >= '2019-01-01')
#                      & (df2019['date'] < '2019-06-01')]
df2020 = pd.read_pickle('Blobs2020df')
df2020.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
# June2020 =df2020.loc[(df2020['date'] >= '2020-01-01')
#                      & (df2020['date'] < '2020-06-01')]
df2021 = pd.read_pickle('Blobs2021df')
df2021.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
# June2021 =df2021.loc[(df2021['date'] >= '2021-01-01')
#                      & (df2021['date'] < '2021-06-01')]
df2022 = pd.read_pickle('Blobs2022df')
df2022.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
# June2022 =df2022.loc[(df2022['date'] >= '2022-01-01')
#                      & (df2022['date'] < '2022-06-01')]

# noi2018 = df2018.groupby(['date'])
# noi2019 = df2019.groupby(['date'])
# noi2020 = df2020.groupby(['date'])
# noi2021 = df2021.groupby(['date'])
# noi2022 = df2022.groupby(['date'])
#
# print('2018',max(df2018.count()),len(noi2018['date']),
#       '2019',max(df2019.count()),len(noi2019['date']),
#       '2020',max(df2020.count()),len(noi2020['date']),
#       '2021',max(df2021.count()),len(noi2021['date']),
#       '2022',max(df2022.count()),len(noi2022['date'])
#       )
#
#
# print('2018',max(June2018.count()),
#       '2019',max(June2019.count()),
#       '2020',max(June2020.count()),
#       '2021',max(June2021.count()),
#       '2022',max(June2022.count())
#       )

# print(df2021['Tile'].value_counts())

data = pd.concat([df2018,df2019,df2020,df2021,df2022], ignore_index=True)
# data.to_pickle('Blobs18-22df')
# data.to_csv('blobs2018-2022')
# data = pd.read_pickle('Blobs18-22df')
# print(data['date'].value_counts())
# data.groupby(data['date']).size().plot()
dates = pd.DataFrame(data['date'].value_counts().reset_index())
dates.columns = ['date','count']
print(dates.head())
dates = dates.sort_values('date',ascending=True)
# plt.plot(dates['date'],dates['count'])
# plt.xticks(rotation='vertical')
# # nob = data['date'].value_counts().sort_index()
# # nob.plot()
# # nob.plot.scatter(x='Date',y='Number of blobs')
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
ax = plt.gca()
ax.plot(dates['date'].drop_duplicates(),dates['count'])
ax.set(xlabel='Date',ylabel='Number of Blobs')
date_form = DateFormatter('%Y-%b')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(date_form)
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

# dates.plot(dates['date'],dates['count'])

# import matplotlib.dates as mdates
# ax = plt.gca()
# # start by your date and then your data
# ax.plot(dates['date'], dates['count'])  # daily data
# # You can change the step of range() as you prefer (now, it selects each third month)
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# # you can change the format of the label (now it is 2016-Jan)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
#
# plt.setp(ax.get_xticklabels(), rotation=90)
# plt.show()