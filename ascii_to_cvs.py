import pandas as pd


##R765RF_D5
R765RF_D5_TT4_name = [1,2,3,4,5,6,7,8,9,10,11,12]
R765RF_D5_TT6_name = [1,2]
R765RF_D5_TT7_name = [1,2,3]
R765RF_D5_TT10_name = [1,2,3,4,5,6,7]
R765RF_D5_TT13_name = [1,2,3]
R765RF_D5_TT15_name = [1,2,3,4,5]

##R781_D2
R781_D2_TT2_name = [1,2,3,4]
R781_D2_TT3_name = [1,2,3,4,5,6,7,8,9]
R781_D2_TT5_name = [1,2,3,4,5,6]
R781_D2_TT9_name = [1,2,3,4]

##R781_D3
R781_D3_TT2_name = [1,2,3,4,5,6]
R781_D3_TT3_name = [1,2,3,4,5,6,7,8]
R781_D3_TT5_name = [1,2,3,4,5,6,7,8]
R781_D3_TT9_name = [1,2]
R781_D3_TT11_name = [1,2]
R781_D3_TT14_name = [1,2]

##R781_D4
R781_D4_TT2_name = [1,2,3,4,5,6,7,8,9]
R781_D4_TT3_name = [1,2,3,4,5,6,7,8]
R781_D4_TT5_name = [1,2,3,4]

#R808_D1
R808_D1_TT1_name = [1,2]
R808_D1_TT6_name = [1,2]
R808_D1_TT9_name = [1]
R808_D1_TT13_name = [1]
R808_D1_TT14_name = [1]
R808_D1_TT15_name = [1,2,3]

#R808_D6
R808_D6_TT9_name = [1,2,3,4]
R808_D6_TT12_name = [1,2,3,4,5,6,7,8,9]
R808_D6_TT13_name = [1,2,3,4,5]

#R808_D7
R808_D7_TT12_name = [1,2,3,4,5,6,7,8,9]
R808_D7_TT15_name = [1,2,3,4]

#R859_D1
R859_D1_TT1_name = [1,2,3,4,5,6]
R859_D1_TT3_name = [1,2,3,4,5,6,7]
R859_D1_TT4_name = [1,2,3,4]
R859_D1_TT5_name = [1,2]
R859_D1_TT6_name = [1,2,3,4,5,6]
R859_D1_TT7_name = [1,2,3,4,5,6,7,8]
R859_D1_TT8_name = [1,2,3,4,5,6,7]
R859_D1_TT14_name = [1,2,3,4,5,6,7,8,9]

#R859_D2
R859_D2_TT1_name = [1,2,3,4,5]
R859_D2_TT6_name = [1,2,3,4,5,6,7,8]
R859_D2_TT7_0001_name = [1,2,3,4,5,6,7,8]
R859_D2_TT8_name = [1,2,3,4,5]
R859_D2_TT10_name = [1,2,3,4,5,6]
R859_D2_TT11_name = [1,2,3,4,5,6,7,8,9,10]
R859_D2_TT12_name = [1,2,3,4]
R859_D2_TT13_name = [1,2,3,4,5]
R859_D2_TT14_name = [1,2,3,4,5,6,7,8]

#R859_D3
R859_D3_TT1_0001_name = [1,2,3,4,5]
R859_D3_TT3_name = [1,2,3,4,5,6]
R859_D3_TT6_name = [1,2,3,4,5,6,7,8,9]
R859_D3_TT7_0001_name = [1,2,3,4,5,6,7,8]
R859_D3_TT8_0001_name = [1,2,3,4]
R859_D3_TT10_name = [1,2,3,4,5,6,7,8]
R859_D3_TT12_name = [1,2,3,4]

#886_D1
R886_D1_TT2_name = [1,2]
R886_D1_TT5_name = [1,2,3]
R886_D1_TT9_name = [1,2,3,4,5]
R886_D1_TT10_name = [1,2]

#886_D2
R886_D2_TT2_name = [1,2]
R886_D2_TT5_name = [1,2,3]
R886_D2_TT10_name = [1,2,3]
R886_D2_TT12_name = [1,2,3,4,5]

#886_D3
R886_D3_TT4_name = [1,2]

for i in R886_D3_TT4_name:
    file =  r'/Users/evashenyueyi/Downloads/city_data/R886_D3/TT4/cl-maze1.' + str(i)

    framefile = pd.read_csv(file, skiprows=8, encoding='gbk', engine='python',sep=',',delimiter=None,skipinitialspace=True)
    # framefile = pd.read_csv(file)

    # print(framefile.columns)
    # print(framefile.iloc[0:,0:])

    framefile.to_csv("/Users/evashenyueyi/Downloads/city_data/R886_D3/TT4/cl-maze1."+ str(i)+".csv",index=False,sep=',')


    