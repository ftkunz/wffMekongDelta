import pandas as pd
import uuid
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from datetime import datetime

import matplotlib.pyplot as plt
#
rd = random.Random()

y = 2021

#Change for different years:
# 0 = 2021, 1 = 2018, 2 = 2019, 3 = 2020, 4 = 2022, 5 = 2016, 6 = 2017
if y == 2016:
    year =5
if y == 2017:
    year = 6
if y == 2018:
    year =1
if y == 2019:
    year = 2
if y == 2020:
    year = 3
if y == 2021:
    year = 0
if y == 2022:
    year = 4
years = [2021,2018,2019,2020,2022,2016,2017] #Sorry weird order but otherwise the unique id for 2021 is not correct after I downloaded a lot of images already

rd.seed(year)
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))


# #DETERMINE THRESHOLD OFFSET - 0.2 WAS CHOSEN!
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

##CREATE DATAFRAME FROM DOWNLOADED CSV FILES

dfblob = []
for file in range(271):
    geodf = pd.read_csv(
        "WWFMekongDelta/Mekong_delta_tables_csv_"+str(years[year])+"/boats_table_csv_Zoom13_th0.2_Tile" + str(file) + "_MekongDelta_"+str(years[year])+".csv")
    geodf['Tile'] = file
    dfblob.append(geodf)
dfblob = pd.concat(dfblob)
dfblob['blobID'] = dfblob.apply(lambda x: uuid.uuid4(), axis=1)
dfblob.to_csv('Blobs'+str(years[year])+'.csv')
#
df = pd.read_csv('Blobs'+str(years[year])+'.csv')
df.to_pickle('Blobs'+str(years[year])+'df')



#VALIDATE DIFFERENT AREA DEFINED FOR ZOOM 13 ('projects/mekongdeltares/assets/zoom13tilesMD')
validate = False
provinces = True
sights = False
if validate:
    VMD = [21, 23, 24, 30, 31, 32, 33, 38,39, 40, 41, 51, 52, 53, 54, 55, 63, 64, 65, 66, 68, 69,
           70,
           71, 74, 75, 76, 77, 78, 79, 80,
           81, 82, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 100, 101, 102, 109, 110, 111, 112, 115, 116, 117, 118, 119,
           122,
           123, 124, 126, 127, 128, 130, 131, 132,
           134, 135, 136, 137, 138, 140, 141, 142, 144, 145, 146, 147, 149, 150, 152, 153, 154, 158, 159, 160, 161, 162,
           173, 174, 175, 176, 177, 179, 180, 181,
           182, 183]
    for i in np.arange(185, 271):
        VMD.append(i)

    SaDec2VinhLong = [134, 135, 136, 140, 144, 149, 152]

    KampongCham = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 34, 35,
                   36,
                   37, 38, 39, 40,
                   42, 43, 44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61, 62, 67]

    AnGiang = [21, 22, 23, 24, 30, 31, 32, 33, 39, 40, 41, 51, 52, 53, 54, 55, 64, 65, 66, 70, 71, 76, 77, 78, 79, 80,
               81,
               82,
               87, 88, 91, 92, 93, 94, 95, 100, 109, 110, 115, 116]
    DongThap = [39, 40, 51, 52, 63, 64, 68, 69, 74, 75, 76, 77, 78, 79, 84, 85, 86, 91, 92, 100, 101, 102, 109, 110,
                111,
                112,
                115, 116, 117, 118, 119, 122, 123, 124, 126, 127, 128, 130, 131, 134, 135, 136]
    CanTho = [101, 102, 111, 112, 118, 119, 123, 124, 127, 128, 132, 137, 138, 141, 142]
    HauGiang = [146, 147, 150]
    VinhLong = [132, 137, 138, 141, 142, 145, 146, 147, 149, 150, 152, 153, 154, 158, 159, 160, 161, 174, 175, 180, 186,
                193, 194, 202, 203, 211, 212]
    TraVinh = [162, 176, 177, 181, 182, 187, 188, 195, 196, 204, 205, 213, 214, 220]
    SocTrang = [150, 154, 161, 162, 176, 177, 181, 182, 183, 189, 190, 197, 198, 206, 207, 208, 215]
    TienGiang = [144, 149, 152, 158, 173, 174, 179, 185, 191, 199, 209, 216, 221, 226, 232, 233, 239, 245,
                 246, 252, 253, 256, 257, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270]


    if provinces:
        provinces = [VMD,SaDec2VinhLong, KampongCham, AnGiang, DongThap, CanTho, HauGiang, VinhLong, TraVinh, SocTrang,
                     TienGiang]
        names = ['VMD','SaDec2VinhLong', 'KampongCham', 'AnGiang', 'DongThap', 'CanTho', 'HauGiang', 'VinhLong', 'TraVinh',
                 'SocTrang', 'TienGiang']

    a = [31,30,23]
    b = [39,40]
    c = [68,69,74,75,63,64]
    d = [76]
    e = [82,87]
    f = [100,109,110,115,116]
    g = [102,111]
    h = [135]
    i = [144,149]
    j = [158,173]
    k = [179,185]
    l = [175,180]
    m = [199,209]
    n = [257,261,262]
    o = [193,194]
    p = [128,132]
    q = [154]
    r = [182]
    s = [190,197,198]
    t = [201,210,217]
    u = [228,229,234]

    if sights:
        provinces = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u]
        names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']

    for i in range(len(names)):

        df = pd.read_pickle('Blobs'+str(years[year])+'df')
        df.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)

        df['DateTime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(int(x)/1000))
        # print(df['DateTime'].drop_duplicates())
        # file1 = open("DateTime", "a")
        # file1.close()
        # timestamps = list(df['timestamp'])
        # dtlist = []
        # for i in range(len(timestamps)):
        #     dtlist.append(datetime.fromtimestamp(int(timestamps[i])))
        # print(dtlist)
        region = names[i]
        # df = df[df['Tile'].isin(SaDec2VinhLong)]
        # df = df[df['Tile'].isin(KampongCham)]
        # print(provinces[i])
        df = df[df['Tile'].isin(provinces[i])]
        # df = df[df['Tile'].isin([85,86])]
        # for t in VMD:
        #     # dft = df[df['Tile']==t]
        # dft = df.loc[(df['date'] >= '2022-03-31')
        #     & (df['date'] < '2022-04-01')]
        # print('tiles',dft['Tile'].drop_duplicates())

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
        # print(df['blobID'])
        # print(df['date'])
        # print(df['date'].value_counts())
        # print(df['Tile'].value_counts())
        # NoB = pd.DataFrame(df['date'].value_counts())
        # print(NoB)
        # NoB.to_csv('NumberofBlobs'+str(years[year])+'.csv')
        # df.hist()
        # df.hist('distance2shore',bins=100)
        # df.hist('area',bins=100)
        # df.hist('width',bins=100)
        # df.hist('height',bins=100)
        # plt.show()
        area = df.groupby(['Tile','date']).agg({'area': 'sum'}).reset_index()
        area['days'] = pd.to_datetime(area['date']).diff(-1)
        # print(area['days'])

        nbc = 0
        listtile = list(area['Tile'].drop_duplicates())

        for tile in listtile:
            TileAbc= area[area['Tile'] == tile]
            startbc = list(TileAbc['area'])[0] * (
                        pd.to_datetime(list(TileAbc['date'])[0]).to_pydatetime().date() - date(years[year], 1, 1)).days
            if len(list(TileAbc['date'])) > 1:
                TileAbc['numberBC'] = -(TileAbc['days'].dt.days) * TileAbc['area']
                bc = TileAbc['numberBC'][:-1].sum(min_count=1)
                if year == 4:
                    endbc = list(TileAbc['area'])[-1] * (date(years[year], 6, 1) - pd.to_datetime(
                        list(TileAbc['date'])[-1]).to_pydatetime().date()).days
                if year != 4:
                    endbc = list(TileAbc['area'])[-1] * (date(years[year], 12, 31) - pd.to_datetime(
                        list(TileAbc['date'])[-1]).to_pydatetime().date()).days

            if len(list(TileAbc['date'])) == 1:
                bc = 0
                if year == 4:
                    endbc = list(TileAbc['area'])[0] * (date(years[year], 6, 1) - pd.to_datetime(
                        list(TileAbc['date'])[0]).to_pydatetime().date()).days
                if year != 4:
                    endbc = list(TileAbc['area'])[0] * (date(years[year], 12, 31) - pd.to_datetime(
                        list(TileAbc['date'])[0]).to_pydatetime().date()).days

            totbc = (startbc + bc + endbc)*7*0.39*0.57
            nbc = nbc + totbc
        print(region,'*method 1',nbc)
        file1 = open("estimate.txt", "a")
        file1.write(str(region)+' *method 1 '+str(nbc)+' \n')
        file1.close()
        area1 = df.groupby(['date']).agg({'area': 'sum'}).reset_index()
        area1['days'] = pd.to_datetime(area1['date']).diff(-1)
        # print(area['days'])
        area1['volume'] = area1['area'] * 7 * 0.39
        area1['msand']=area1['volume']
        area1['Msand']=area1['msand']*-(area['days'].dt.days)
        corr = 0.57
        print(region,'method 1',corr*area1['Msand'].sum(min_count=1))
        # print(area)

        dfbc = dfold[dfold['distance2shore']>50]
        # print('tiles2021',dfbc['Tile'].value_counts())
        dfbc = dfbc[dfbc['height']>30]
        dfbc = dfbc[dfbc['height']<44]
        dfbc = dfbc[dfbc['area']>352]
        dfbc = dfbc[dfbc['area']<548]

        Abc = dfbc.groupby(['Tile','date']).agg({'area': 'sum'}).reset_index()

        getCount = dfbc.groupby(['Tile','date']).size().reset_index(name='count')
        Abc['count'] = getCount['count']

        nbc = 0
        Abc['days'] = pd.to_datetime(Abc['date']).diff(-1)
        listtile = list(Abc['Tile'].drop_duplicates())

        for tile in listtile:
            TileAbc = Abc[Abc['Tile']==tile]
            # print('test',TileAbc)
            # print(-(TileAbc['days'].dt.days) * TileAbc['count'])
            # print(Abc['date'][0])
            startbc = list(TileAbc['count'])[0]*(pd.to_datetime(list(TileAbc['date'])[0]).to_pydatetime().date()-date(years[year], 1, 1)).days

            if len(list(TileAbc['date'])) > 1:
                TileAbc['numberBC'] = -(TileAbc['days'].dt.days) * TileAbc['count']
                bc = TileAbc['numberBC'][:-1].sum(min_count=1)
                if year == 4:
                    endbc = list(TileAbc['count'])[-1] * (date(years[year], 6, 1)-pd.to_datetime(list(TileAbc['date'])[-1]).to_pydatetime().date()).days
                if year != 4:
                    endbc = list(TileAbc['count'])[-1] * (date(years[year], 12, 31)-pd.to_datetime(list(TileAbc['date'])[-1]).to_pydatetime().date()).days

            if len(list(TileAbc['date'])) == 1:
                bc = 0
                if year == 4:
                    endbc = list(TileAbc['count'])[0] * (date(years[year], 6, 1)-pd.to_datetime(list(TileAbc['date'])[0]).to_pydatetime().date()).days
                if year != 4:
                    endbc = list(TileAbc['count'])[0] * (date(years[year], 12, 31)-pd.to_datetime(list(TileAbc['date'])[0]).to_pydatetime().date()).days

            totbc = startbc + bc + endbc
            nbc = nbc +totbc


        print('nbc',nbc/365)
        print('*method3',nbc*300,nbc*450,nbc*600)
        file1 = open("estimate.txt", "a")
        file1.write(str(region)+ ' *method3 '+str(nbc*300)+' ' +str(nbc*450) +' ' +str(nbc*600)+' \n')
        file1.close()

        # print(Abc)
        # print(len(Abc['count']),Abc['count'])
        Abc1 = dfbc.groupby(['date']).agg({'area': 'sum'}).reset_index()
        getCount1 = dfbc.groupby(['date']).size().reset_index(name='count')
        Abc1['count'] = getCount1['count']
        Abc1['days'] = pd.to_datetime(Abc1['date']).diff(-1)
        Abc1['numberBC'] =-(Abc1['days'].dt.days)*Abc1['count']
        Abc1['volume300'] = -(Abc1['days'].dt.days)*Abc1['count']*300
        Abc1['volume600'] = -(Abc1['days'].dt.days)*Abc1['count']*600
        Abc1['volume']=Abc1['area']*7*0.39*0.3
        Abc1['msand']=Abc1['volume']
        Abc1['Msand']=Abc1['msand']*-(Abc['days'].dt.days)
        Mbc = Abc1['Msand'].sum(min_count=1)
        print('bc',Abc1['numberBC'].sum(min_count=1))
        print(region,'method 3',Abc1['volume300'].sum(min_count=1),Abc1['volume600'].sum(min_count=1))

        # Abc = Abc.loc[(Abc['date'] >= '2022-03-30')
        #                   & (Abc['date'] < '2022-04-15')]
        # print('april',Abc['date'],Abc['count'])

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

        # print((Mbc+Mbt+Mbb))

        # print((Mbc+Mb)*1600)

        dfbc['area'] = 0.33*dfbc['area']
        DF = pd.concat([dfbb,dfbt,dfbc], ignore_index=True)
        DF.drop_duplicates(subset= ['timestamp','cenlon','cenlat'],ignore_index=True)
        # print(DF)

        A = DF.groupby(['Tile','date']).agg({'area': 'sum'}).reset_index()
        A['days'] = pd.to_datetime(A['date']).diff(-1)

        nb = 0
        listtile = list(A['Tile'].drop_duplicates())

        for tile in listtile:
            TileA = A[A['Tile'] == tile]
            startb = list(TileA['area'])[0] * (
                    pd.to_datetime(list(TileA['date'])[0]).to_pydatetime().date() - date(years[year], 1, 1)).days
            if len(list(TileA['date'])) > 1:
                TileA['numberBC'] = -(TileA['days'].dt.days) * TileA['area']
                b = TileA['numberBC'][:-1].sum(min_count=1)
                if year == 4:
                    endb = list(TileA['area'])[-1] * (date(years[year], 6, 1) - pd.to_datetime(
                        list(TileA['date'])[-1]).to_pydatetime().date()).days
                if year != 4:
                    endb = list(TileA['area'])[-1] * (date(years[year], 12, 31) - pd.to_datetime(
                        list(TileA['date'])[-1]).to_pydatetime().date()).days

            if len(list(TileA['date'])) == 1:
                bc = 0
                if year == 4:
                    endb = list(TileA['area'])[0] * (date(years[year], 6, 1) - pd.to_datetime(
                        list(TileA['date'])[0]).to_pydatetime().date()).days
                if year != 4:
                    endb = list(TileA['area'])[0] * (date(years[year], 12, 31) - pd.to_datetime(
                        list(TileA['date'])[0]).to_pydatetime().date()).days

            totb = (startb + b + endb) * 7 * 0.39
            nb = nb + totb
        print(region, '*method 2', nb)
        file1 = open("estimate.txt", "a")
        file1.write(str(region)+ ' *method 2 '+str(nb)+' \n')
        file1.close()

        Aold = DF.groupby(['Tile','date']).agg({'area': 'sum'}).reset_index()
        Aold['days'] = pd.to_datetime(Aold['date']).diff(-1)
        Aold['volume']=Aold['area']*7*0.39
        Aold['msand']=Aold['volume']
        Aold['Msand']=Aold['msand']*-(Aold['days'].dt.days)
        M = Aold['Msand'].sum(min_count=1)

        print(region,'method 2',M)


##MAKE HISTOGRAMS
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

##GET DATASET WITH DETECTED BARGES WITH CRANE
barges_with_crane_df = False
if barges_with_crane_df:
    VMD = [21, 23, 24, 30, 31, 32, 33, 38,39, 40, 41, 51, 52, 53, 54, 55, 63, 64, 65, 66, 68, 69,
               70,
               71, 74, 75, 76, 77, 78, 79, 80,
               81, 82, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 100, 101, 102, 109, 110, 111, 112, 115, 116, 117, 118, 119,
               122,
               123, 124, 126, 127, 128, 130, 131, 132,
               134, 135, 136, 137, 138, 140, 141, 142, 144, 145, 146, 147, 149, 150, 152, 153, 154, 158, 159, 160, 161, 162,
               173, 174, 175, 176, 177, 179, 180, 181,
               182, 183]

    # df2018 = pd.read_pickle('Blobs2018df')
    # df2018 = df2018[df2018['Tile'].isin(VMD)]
    # df2018.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    # df2018= df2018[df2018['distance2shore']>80]
    # df2018 = df2018[df2018['height'] > 30]
    # df2018 = df2018[df2018['height'] < 44]
    # df2018 = df2018[df2018['area'] > 352]
    # df2018 = df2018[df2018['area'] < 548]
    # df2018.to_csv('bc2018')
    # print(df2018['Tile'].value_counts())
    #
    # # June2018 =df2018.loc[(df2018['date'] >= '2018-01-01')
    # #                      & (df2018['date'] < '2018-06-01')]
    # df2019 = pd.read_pickle('Blobs2019df')
    # df2019 = df2019[df2019['Tile'].isin(VMD)]
    # df2019.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    # df2019= df2019[df2019['distance2shore']>80]
    # df2019 = df2019[df2019['height'] > 30]
    # df2019 = df2019[df2019['height'] < 44]
    # df2019 = df2019[df2019['area'] > 352]
    # df2019 = df2019[df2019['area'] < 548]
    # df2019.to_csv('bc2019')
    # print(df2019['Tile'].value_counts())
    # # June2019 =df2019.loc[(df2019['date'] >= '2019-01-01')
    # #                      & (df2019['date'] < '2019-06-01')]
    df2020 = pd.read_pickle('Blobs2020df')
    df2020 = df2020[df2020['Tile'].isin(VMD)]
    df2020.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    df2020= df2020[df2020['distance2shore']>80]
    df2020 = df2020[df2020['height'] > 30]
    df2020 = df2020[df2020['height'] < 44]
    df2020 = df2020[df2020['area'] > 352]
    df2020 = df2020[df2020['area'] < 548]
    df2020.to_csv('bc2020')
    df2020.to_pickle('bc2020df')
    print(df2020['Tile'].value_counts())
    # June2020 =df2020.loc[(df2020['date'] >= '2020-01-01')
    #                      & (df2020['date'] < '2020-06-01')]
    df2021 = pd.read_pickle('Blobs2021df')
    df2021 = df2021[df2021['Tile'].isin(VMD)]
    df2021.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    df2021= df2021[df2021['distance2shore']>80]
    df2021 = df2021[df2021['height'] > 30]
    df2021 = df2021[df2021['height'] < 44]
    df2021 = df2021[df2021['area'] > 352]
    df2021 = df2021[df2021['area'] < 548]
    df2021.to_csv('bc2021')
    df2021.to_pickle('bc2021df')
    print(df2021['Tile'].value_counts())
    # June2021 =df2021.loc[(df2021['date'] >= '2021-01-01')
    #                      & (df2021['date'] < '2021-06-01')]


    df2022 = pd.read_pickle('Blobs2022df')
    df2022 = df2022[df2022['Tile'].isin(VMD)]
    df2022.drop_duplicates(subset=['timestamp','cenlat','cenlon'],keep='first',ignore_index=True)
    df2022= df2022[df2022['distance2shore']>80]
    df2022 = df2022[df2022['height'] > 30]
    df2022 = df2022[df2022['height'] < 44]
    df2022 = df2022[df2022['area'] > 352]
    df2022 = df2022[df2022['area'] < 548]
    df2022.to_csv('bc2022')
    df2022.to_pickle('bc2022df')
    print(df2022['Tile'].value_counts())

    # df2020 = pd.read_pickle('bc2022df')
    # df2021 = pd.read_pickle('bc2022df')
    # df2022 = pd.read_pickle('bc2022df')
    data = pd.concat([df2020,df2021,df2022], ignore_index=False)
    BC = data.groupby(['Tile','date']).agg({'area': 'sum'}).reset_index()
    getCount = data.groupby(['Tile', 'date']).size().reset_index(name='count')
    BC['count'] = getCount['count']

    plt.scatter(BC['date'],BC['count'],c=BC['Tile'])
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

    # data = pd.read_pickle('bc2022df')
    data['date'] = pd.to_datetime(data['date'])
    BC = data.groupby(['date']).agg({'area': 'sum'}).reset_index()
    getCount = data.groupby(['date']).size().reset_index(name='count')
    BC['count'] = getCount['count']


    fig,ax2 = plt.subplots()
    max = BC['count']*600
    min = BC['count']*300
    mean = BC['count']*450
    ax2.plot(BC['date'],mean, color = 'darkorange')
    ax2.plot(BC['date'],max, color = 'orange')
    ax2.plot(BC['date'],min, color = 'orange')
    ax2.set_ylabel('Extracted Volume [m3/d]',color='darkorange',fontsize=14)
    # ax2.set_xticklabels(BC['date'],rotation='vertical')
    ax2.set_xlim(list(BC['date'])[0],list(BC['date'])[-1])
    ax2.fill_between(BC['date'],min,max,color = 'orange')
    ax = ax2.twinx()
    ax.scatter(BC['date'],BC['count'], color = 'black')
    ax.set_xlabel('date',fontsize=14)
    ax.set_ylabel('Number of BC',color = 'black', fontsize = 14)

    # plt.fill_between(min,max,color='yellow')
    # ax.xticks(rotation='vertical')
    plt.show()

# June2022 =df2022.loc[(df2022['date'] >= '2022-01-01')
#                      & (df2022['date'] < '2022-06-01')]
#
# # noi2018 = df2018.groupby(['date'])
# # noi2019 = df2019.groupby(['date'])
# # noi2020 = df2020.groupby(['date'])
# # noi2021 = df2021.groupby(['date'])
# # noi2022 = df2022.groupby(['date'])
# #
# # print('2018',max(df2018.count()),len(noi2018['date']),
# #       '2019',max(df2019.count()),len(noi2019['date']),
# #       '2020',max(df2020.count()),len(noi2020['date']),
# #       '2021',max(df2021.count()),len(noi2021['date']),
# #       '2022',max(df2022.count()),len(noi2022['date'])
# #       )
# #
# #
# # print('2018',max(June2018.count()),
# #       '2019',max(June2019.count()),
# #       '2020',max(June2020.count()),
# #       '2021',max(June2021.count()),
# #       '2022',max(June2022.count())
# #       )
#
# # print(df2021['Tile'].value_counts())
#
# data = pd.concat([df2020,df2021,df2022], ignore_index=True)
# # data.to_pickle('Blobs18-22df')
# # data.to_csv('blobs2018-2022')
# # data = pd.read_pickle('Blobs18-22df')
# # print(data['date'].value_counts())
# # data.groupby(data['date']).size().plot()
# dates = pd.DataFrame(data['date'].value_counts().reset_index())
# dates.columns = ['date','count']
# print(dates.head())
# dates = dates.sort_values('date',ascending=True)
# # plt.plot(dates['date'],dates['count'])
# # plt.xticks(rotation='vertical')
# # # nob = data['date'].value_counts().sort_index()
# # # nob.plot()
# # # nob.plot.scatter(x='Date',y='Number of blobs')
# import matplotlib.dates as mdates
# from matplotlib.dates import DateFormatter
# ax = plt.gca()
# ax.plot(dates['date'].drop_duplicates(),dates['count'])
# ax.set(xlabel='Date',ylabel='Number of Blobs')
# date_form = DateFormatter('%Y-%b')
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(date_form)
# plt.setp(ax.get_xticklabels(), rotation=90)
# plt.show()

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