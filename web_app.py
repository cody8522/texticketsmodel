import pandas as pd
import numpy as np
import streamlit as st
import random
import pickle
import zipfile

print('started df formation')


extradroplst = ['Section','Artist Name', 'Name', 'City', 'State','DayOfWeek','Month']
chartcols = ['sp_followers',
	 'sp_popularity',
	 'sp_followers_to_listeners_ratio',
	 'sp_monthly_listeners',
	 'sp_playlist_total_reach',
	 'cm_artist_rank','cm_artist_score','facebook_followers','ins_followers']



@st.experimental_memo(suppress_st_warning = True)
def csv_load():
	df = pd.read_csv('BIG-SHOWS-RAW.csv')
	st.write(df.head())
	event = pd.read_csv('Old Shows - Old Shows (1).csv')
	venue_info = pd.read_csv('Venue Information.csv')
	df = df.merge(venue_info[['Name','Adjusted Capacity']], on='Name').drop('Unnamed: 0',axis=1)
	df = df.merge(event[['Stubhub ID','Month']],on='Stubhub ID',how='left')
	print(len(df))
	mydf = df.groupby(['Stubhub ID','Section']).agg({'DayOfWeek':'count','Get-In':'median'}).sort_values(by='DayOfWeek',ascending=False).reset_index()
	idlist = list(set(list(mydf[mydf['DayOfWeek']<365]['Stubhub ID'])))
	df = df[df['Stubhub ID'].isin(idlist)]
	print(len(df))

	df['Artist Name'] = list(map(lambda x: x.strip().title(), df['Artist Name']))
	chart = pd.read_csv('chartmetric - chartmetric.csv')
	chart['name'] = list(map(lambda x: x.strip().title(), chart['name']))
	tst = chart[['name'] + chartcols].reset_index().dropna().rename(columns={'name':'Artist Name'})
	df = df.merge(tst,on='Artist Name',how='left').dropna()
	return df
csv_load()

df = csv_load()

@st.experimental_memo(suppress_st_warning = True)
def venue_list_func():
	venue_list = df['Name'].tolist()
	venue_list = sorted(list(set(venue_list)))
	return venue_list
venue_list_func()
venue_list = venue_list_func()

@st.experimental_memo(suppress_st_warning = True)
def city_list_func():
	city_list = df['City'].tolist()
	city_list = sorted(list(set(city_list)))
	return city_list
city_list_func()
city_list = city_list_func()

@st.experimental_memo(suppress_st_warning = True)
def artist_list_func():
	artist_list = df['Artist Name'].tolist()
	artist_list = sorted(list(set(artist_list)))
	return artist_list
artist_list_func()
artist_list = artist_list_func()


@st.experimental_memo(suppress_st_warning = True)
def sl_2():
	groupvars = ['Artist Name','Section','DayOfWeek','Month','Name','Adjusted Capacity','City','State','Cons1','Cons2']
	mldfpre = df.groupby(groupvars).median()[['Get-In'] + chartcols].reset_index()
	mldf = df.groupby(groupvars).median()[['Get-In'] + chartcols].reset_index()

	encode_dict = {}
	for z in extradroplst:

	    grouplst = mldf.groupby(z).median().sort_values(by='Get-In', ascending=False)[['Get-In']]

	    my_dict = {}
	    
	    for i,j in grouplst['Get-In'].items():
	        if j not in list(my_dict.values()):
	            my_dict[i]=j
	        else:
	#            print('test')
	#            print(j)
	            my_dict[i]=j+0.1

	    random.seed(0)
	    subdf = pd.DataFrame(my_dict,index=range(1)).T
	    subdf[0] = [random.random()/100 + i for i in list(my_dict.values())]
	    my_dict = dict(dict(subdf)[0])
	    
	    mldf[z] = list(map(lambda x: my_dict[x] if x in my_dict else 9999, mldf[z]))

	    print('SECTION ENCODING COMPLETE')

	    try:
	        mldf = mldf.drop(['index'], axis=1)   
	    except:
	        pass
	    
	    encode_dict[z]=my_dict
	    grouplst.to_csv('grouplst.csv')

	mldf = mldf.rename(columns={'Name':'Venue'})


	merger = mldf.groupby(['Artist Name','Venue']).median().reset_index()[['Artist Name','Venue','Get-In']].rename(columns={'Get-In':'Artist-Venue'})
	output = open('Artist-Venue.pkl', 'wb')
	pickle.dump(merger, output)

	mldf = mldf.merge(merger,on=['Artist Name','Venue'],how='left')

	merger = mldf.groupby(['Artist Name','Section']).median().reset_index()[['Artist Name','Section','Get-In']].rename(columns={'Get-In':'Artist-Section'})
	output = open('Artist-Section.pkl', 'wb')
	pickle.dump(merger, output)
	mldf = mldf.merge(merger,on=['Artist Name','Section'],how='left')

	merger = mldf.groupby(['Venue','Section']).median().reset_index()[['Venue','Section','Get-In']].rename(columns={'Get-In':'Venue-Section'})
	output = open('Venue-Section.pkl', 'wb')
	pickle.dump(merger, output)
	mldf = mldf.merge(merger,on=['Venue','Section'],how='left')

	merger = mldf.groupby(['Artist Name','Venue','Section']).median().reset_index()[['Artist Name','Venue','Section','Get-In']].rename(columns={'Get-In':'Artist-Venue-Section'})
	output = open('Artist-Venue-Section.pkl', 'wb')
	pickle.dump(merger, output)
	mldf = mldf.merge(merger,on=['Artist Name','Venue','Section'],how='left')[['Artist Name','Venue','Section','Artist-Venue','Artist-Section','Venue-Section','Artist-Venue-Section','DayOfWeek', 'Month',
	       'Adjusted Capacity', 'City', 'State', 'Cons1', 'Cons2', 'Get-In'] + chartcols]

	output = open('encode_dict.pkl', 'wb')
	pickle.dump(encode_dict, output)
	print('done')
	return mldf
sl_2()
mldf = sl_2()



# stublst = list(df['Stubhub ID'].value_counts().keys())
# random.shuffle(stublst)
# stublst = stublst[:int(len(stublst)*.90)]
# dfholdout = df[~df['Stubhub ID'].isin(stublst)]
# dfholdout.to_csv('dfholdout.csv')
# df = df[df['Stubhub ID'].isin(stublst)]





@st.experimental_memo(suppress_st_warning = True)
def prevshows(artistvar,venuevar,sectionvar,eventfull,mydisplay=True):
    chartcols = ['sp_followers',
 'sp_popularity',
 'sp_followers_to_listeners_ratio',
 'sp_monthly_listeners',
 'sp_playlist_total_reach',
 'cm_artist_rank','cm_artist_score','facebook_followers','ins_followers']
    
    if mydisplay:
        display(df[(df['Name']==venuevar)&(df['Section']==sectionvar)].groupby('Artist Name').min()[['Get-In']].sort_values(by='Get-In').rename(columns={'Get-In': venuevar+'-'+sectionvar+'-'+'Get-In'}))
    
    pd.options.display.max_rows=100
    testdf = df[df['Artist Name']==artistvar]
    testdf = testdf[testdf['Name']==venuevar]
    if sectionvar:
        testdf = testdf[testdf['Section']==sectionvar]
        if len(testdf)==0:
            testdf = df[df['Artist Name']==artistvar]
            testdf = testdf[testdf['Name']==venuevar]

    if len(testdf)>0:
        a = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).min()[['Get-In']].reset_index().rename(columns={'Get-In':'Min Get-In'})
        b = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).median()[['Get-In']].reset_index().rename(columns={'Get-In':artistvar+'-'+venuevar+' Median Get-In'})
        c = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).max()[['Get-In']].reset_index().rename(columns={'Get-In':'Max Get-In'})
        x = a.merge(b).merge(c).sort_values(by=artistvar+'-'+venuevar+' Median Get-In')
#        display(x)    

    pd.options.display.max_rows=100
    testdf = df[df['Artist Name']==artistvar]
    if sectionvar:
        testdf = testdf[testdf['Section']==sectionvar]
        if len(testdf)==0:
            testdf = df[df['Artist Name']==artistvar]
    if len(testdf)>0:
        a = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).min()[['Get-In']].reset_index().rename(columns={'Get-In':'Min Get-In'})
        b = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).median()[['Get-In']].reset_index().rename(columns={'Get-In':artistvar + " Median Get-In"})
        c = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).max()[['Get-In']].reset_index().rename(columns={'Get-In':'Max Get-In'})
        y = a.merge(b).merge(c).sort_values(by=artistvar + ' Median Get-In')

#        display(y)

    pd.options.display.max_rows=100
    testdf = df[df['Name']==venuevar]
    if sectionvar:
        testdf = testdf[testdf['Section']==sectionvar]
        if len(testdf)==0:
            testdf = df[df['Name']==venuevar]
    if len(testdf)>0:

        a = testdf.groupby(["Artist Name",'Name','Adjusted Capacity','City','State',"Stubhub ID",'Section']).min()[['Get-In']].reset_index().rename(columns={'Get-In':'Min Get-In'})
        b = testdf.groupby(["Artist Name",'Name','Adjusted Capacity','City','State',"Stubhub ID",'Section']).median()[['Get-In']].reset_index().rename(columns={'Get-In':venuevar + " Median Get-In"})
        c = testdf.groupby(["Artist Name","Name",'Adjusted Capacity','City','State',"Stubhub ID",'Section']).max()[['Get-In']].reset_index().rename(columns={'Get-In':'Max Get-In'})
        z = a.merge(b).merge(c).sort_values(by= venuevar + " Median Get-In")

#        display(z)

    try:
        x2 = pd.DataFrame(x[artistvar+'-'+venuevar+' Median Get-In'].describe()).T[['min','25%','50%','75%','max','count']]
        y2 = pd.DataFrame(y[artistvar + " Median Get-In"].describe()).T[['min','25%','50%','75%','max','count']]
        z2 = pd.DataFrame(z[venuevar + " Median Get-In"].describe()).T[['min','25%','50%','75%','max','count']]
        if mydisplay:
            display(pd.concat([x2,y2,z2]))
    except:
        try:
            y2 = pd.DataFrame(y[artistvar + " Median Get-In"].describe()).T[['min','25%','50%','75%','max','count']]
            z2 = pd.DataFrame(z[venuevar + " Median Get-In"].describe()).T[['min','25%','50%','75%','max','count']]
            if mydisplay:
                display(pd.concat([y2,z2]))
        except:
            try:
                y2 = pd.DataFrame(y[artistvar + " Median Get-In"].describe()).T[['min','25%','50%','75%','max','count']]
                if mydisplay:
                    display(y2)
            except:
                pass
    
    event = eventfull.copy()        
        
        
    event = event.merge(pd.read_csv('Venue Information.csv').rename(columns={'Venue':'Name'})[['Name','Adjusted Capacity']],on='Name',how='left')

    
    chart = pd.read_csv('chartmetric - chartmetric.csv')
    chart['name'] = list(map(lambda x: x.strip().title(), chart['name']))
    chart = chart[['name']+chartcols].dropna().rename(columns={'name':'Artist Name'})
    
    event = event.merge(chart,on='Artist Name',how='left')
    
    
    for c in ['Artist Name','Name','City','State','DayOfWeek','Month','Section']:
        
        try:
            event[c] = pickle.load(open('encode_dict.pkl','rb'))[c][event[c].iloc[0]]
        except:
            if mydisplay:
                print(c)
            event[c] = event['Artist Name'].iloc[0]
            if mydisplay:
                print('Imputed ' + c + ' To Artist Average')
        

    event = event.rename(columns={'Name':'Venue'})

    eventmerge = event.copy()

    eventmerge = eventmerge.merge(pickle.load(open('Artist-Section.pkl','rb')),how='left',on=['Artist Name','Section'])
    eventmerge = eventmerge.merge(pickle.load(open('Artist-Venue.pkl','rb')),how='left',on=['Artist Name','Venue'])
    eventmerge = eventmerge.merge(pickle.load(open('Venue-Section.pkl','rb')),how='left',on=['Venue','Section'])
    eventmerge = eventmerge.merge(pickle.load(open('Artist-Venue-Section.pkl','rb')),how='left',on=['Artist Name','Venue','Section'])
    eventmerge = eventmerge[list(mldf.drop('Get-In',axis=1).columns)]
    

    
#     for c in event.columns:
#         eventmerge[c] = eventmerge[c].fillna(20554)

    
    varlst = ['All','Artist-Venue','Artist-Section','Venue-Section','Artist-Venue-Section']
    
    #display(eventmerge)
    
    predstree = {}
    predslinear = {}
    
    for i in varlst:
        if i!='All' and np.isnan(eventmerge.iloc[0][i]):
            pass
        else:
            model = pickle.load(open('model_dict.sav','rb'))[i]
            droplst = [q for q in varlst if q!=i]
            droplst = [q for q in droplst if q!='All']
            if i!='Artist-Venue-Section':
                test = eventmerge.drop(droplst,axis=1)
            else:
                test = eventmerge.copy()
            predstree[i] = model.predict(test.values)[0]
            
#             model = pickle.load(open('model_dict_linear.sav','rb'))[i]
#             droplst = [q for q in varlst if q!=i]
#             droplst = [q for q in droplst if q!='All']
#             if i!='Artist-Venue-Section':
#                 test = eventmerge.drop(droplst,axis=1)
#             else:
#                 test = eventmerge.copy()
#             predslinear[i] = model.predict(test.values)[0]

    droplst = [i for i in droplst if i!='All'] + [i if i!='Name' else 'Venue' for i in extradroplst]
#    model = pickle.load(open('model_dict_test.sav','rb'))['Chartmetric/Demographic Only']
    droplst = [q for q in varlst if q!='All']
    test = eventmerge[['Adjusted Capacity',
 'Cons1',
 'Cons2',
 'sp_followers',
 'sp_popularity',
 'sp_followers_to_listeners_ratio',
 'sp_monthly_listeners',
 'sp_playlist_total_reach',
 'cm_artist_rank',
 'cm_artist_score',
 'facebook_followers',
 'ins_followers']]
#    predstree['Chartmetric/Demographic Only'] = model.predict(test.values)[0]
#    model = pickle.load(open('model_dict_linear_test.sav','rb'))['Chartmetric/Demographic Only']
#    predslinear['Chartmetric/Demographic Only'] = model.predict(test.values)[0]
    
    import statistics as st
    
    findf = pd.concat([pd.DataFrame(predstree,index=range(1)).T.rename(columns={0:'Tree Based'}),pd.DataFrame(predslinear,index=range(1)).T.rename(columns={0: 'Linear'})],axis=1)
    if mydisplay:
        display(findf)
    try:
        pred = findf.loc['Artist-Venue-Section'].mean()
        print('Artist-Venue-Section Model')
        return [pred, [pred - pred*0.15, pred + pred*0.15],[pred - pred*0.35, pred + pred*0.35],[pred - pred*0.50, pred + pred*0.50]]
    except:
        try:
            pred = st.mean(list(findf.drop(['All'])['Tree Based']))
            print('Two-Combo Model')
            return [pred, [pred - pred*0.20, pred + pred*0.20],[pred - pred*0.40, pred + pred*0.40],[pred - pred*0.625, pred + pred*0.625]]
        except:
            pred = findf.loc['All'].mean()
            print('Individual Model')
            return [pred, [pred - pred*0.275, pred + pred*0.275],[pred - pred*0.45, pred + pred*0.45],[pred - pred*0.65, pred + pred*0.65]]

@st.experimental_memo(suppress_st_warning = True)       
def showmodel(df, venue, artist, month, DayOfWeek, city, state, cons1, cons2, displaybool=True):

    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    import pandas as pd
    import warnings
    from os import listdir
    from os.path import isfile, join
    import ast
    from sklearn import preprocessing
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    import random
    import requests
    import json
    import datetime
    from datetime import datetime
    import numpy as np
    from sklearn import preprocessing
    from bs4 import BeautifulSoup
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.ensemble import RandomForestRegressor

    venuevar = venue
    artistvar = artist
    mydict = {}
    for i in list(df[df['Name']==venuevar]['Section'].value_counts().keys()):
        eventfull = pd.DataFrame({'Artist Name': artist,
         'Section': i,#'Balcony Center',
         'Month': month,
         'DayOfWeek': DayOfWeek,
         'Name': venuevar,
         'City': city,
         'State': state,
         'Cons1': cons1,
         'Cons2': cons2},index=range(1))

        artistvar = eventfull['Artist Name'].iloc[0]
        venuevar = eventfull['Name'].iloc[0]
        sectionvar = eventfull['Section'].iloc[0]
        a = prevshows(artistvar,venuevar,sectionvar,eventfull,mydisplay=displaybool)
        mydict[i] = a
        
    findf = pd.DataFrame(mydict,index=range(4)).T.rename(columns={0:artistvar+'-'+venuevar+' Prediction',1:'50% Confidence Interval', 2:'75% Confidence Interval',3:'90% Confidence Interval'})
    for c in findf.columns:
        try:
            findf[c] = [round(i,2) for i in list(findf[c])]
        except:
            findf[c] = [[round(i[0],2),round(i[1],2)] for i in findf[c]]
    
    return findf
	
	
import streamlit as st


st.write("""
# TexTickets Fair Market Value App

### Select your parameters for a FMV estimation.

""")
def user_input_features():
	with st.sidebar:
		st.header('User Input Parameters')
		with st.sidebar.form("form"):
		    ArtistName = st.selectbox('Artist Name', artist_list)
		    Month = st.selectbox('Month', ('1','2','3','4','5','6','7','8','9','10','11','12'))
		    DayOfWeek = st.selectbox('DayOfWeek', ('1','2','3','4','5','6','7'))
		    State = st.selectbox('State', ('AL', 'AK', 'AZ', ' AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
		    	'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'))
		    City = st.selectbox('City', city_list)
		    Name = st.selectbox('Venue', venue_list)
		    Section = st.selectbox('Section', ('All', 'None'))
		    Cons1 = st.selectbox('Cons1', ('0', '1'))
		    Cons2 = st.selectbox('Cons2', ('0', '1'))
		    data = {'Artist Name' : ArtistName,
		    'Section' : Section,
		    'Month' : Month,
		    'DayOfWeek' : DayOfWeek,
		    'Name' : Name,
		    'City': City,
		    'State' : State,
		    'Cons1' : Cons1,
		    'Cons2' : Cons2}
		    button_check = st.form_submit_button("Submit")
		    features = pd.DataFrame(data, index = [0])
		    return features
stdf = user_input_features()
st.subheader('User Selections')
st.write(stdf)

###### Model Run

artistvar = stdf['Artist Name'].iloc[0]
venuevar = stdf['Name'].iloc[0]
sectionvar = stdf['Section'].iloc[0]
cityvar = stdf['City'].iloc[0]
statevar = stdf['State'].iloc[0]
monthvar = stdf['Month'].iloc[0]
dayvar = stdf['DayOfWeek'].iloc[0]
cons1var = stdf['Cons1'].iloc[0]
cons2var = stdf['Cons2'].iloc[0]


a = showmodel(df=df, venue=venuevar, artist=artistvar, month=monthvar, DayOfWeek=dayvar, city=cityvar, state=statevar, cons1=cons1var, cons2=cons2var,displaybool=False)
st.write(a)


