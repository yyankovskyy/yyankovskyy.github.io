# -*- coding: utf-8 -*-
"""
Algorithm:
1.	Connect to DB (Input parameters: Server name, DB name); fetch distinct strings from DB
2.	Check for Input parameter Language. If itís Chinese or Russian we translate it to English letters (This is not used in PtP Analyzer; we do only for English)
3.	Clean strings: Replace all non-alphanumeric characters to space
4.	Tokenize the strings word by word. (space is a separator)
5.	If the token is numeric we leave it as is. If itís a string we compute double-metaphone scores
6.	Compare every string with all other string (between which comparison has not happened) and compute the overlap percentage of the double-metaphone scores
7.	If double-metaphone overlap percentage is more than the threshold (Input parameter) then we compute the Jaro edit distance of those two strings
8.	We take greedy approach when one string is matched more than one strings with double-metaphone score more than the input threshold for double-metaphone. We take the match with the highest score, and ignore the rest of the matches
9.	We further filter out the strings that have Jaro distance score less than the input threshold for Jaro edit distance
10.	We push the final result to a table in the DB (using the same database connection)

Constraints:
Currently, we face issues running it for larger number of distinct strings because it exhausts all memory doing comparison between strings in step 6 to calculate overlap percentage score. For n number of distinct strings number of comparisons =  n*(n+1)/2. This is probably an opportunity of improvement (if possible).
After discussing with SMEs, we have restricted this fuzzy logic to happen only on 5000 unique strings, so that it does not eat up all memory in local laptop or fails because of memory out of exceptions.


@author: Eugene.Yankovsky
"""
def save_fuzzy_duplicate (ServerName, DBName, SourceQuery, DestinationTableName, EntityName, PropertyName, Threshold_DM, Threshold_JD, Language ): 
    """
    ServerName : Server name where source/destination data will reside
    DBName: Source database name
    SourceQuery: 
            SQL query to send distinct names/addresses from the table. The SP takes inputs EntityName, PropertyName, NumberOfRecords: 
            exec [dbo].[usp_DM_save_inputlist_fuzzy] 'Vendor', 'Name', 5
    DestinationTableName: "[dbo].[FuzzyResults]" (Change it if you have created the table with a different name in step 2)
    EntityName: To distinguish a test. For example: “Vendor”
    PropertyName: To distinguish address and name. Example: “Name”
    Threshold_DM: (for double metaphone) matching % value for which you want to consider two strings to be duplicate. Example: 80
    Language: "Chinese", "Russian", "French", "English"
    """
    import pyodbc
    import jellyfish as d
    from metaphone import doublemetaphone
    import pandas as pd
    import time
    import re
    import inflect
    from transliterate import translit
    from xpinyin import Pinyin
    p = inflect.engine()
    
    start = time.time()
    r=[]
    t = []
    data = []
    true_data = []
    tokenized = []
    p_final = pd.DataFrame.from_records(t, columns=["sn1", "sn2", "s1", "s2", "m1", "m2", "Overlap", "OverlapPercent", "JaroDist"])
    p_temp = pd.DataFrame.from_records(t, columns=["sn1", "sn2", "s1", "s2", "m1", "m2", "Overlap", "OverlapPercent", "JaroDist"])
    i = 0
    print("Starting to load data from DB...")
    #conn = pymssql.connect(server=IN2174234W2, user=user, password=password, database=db)
    #conn = pyodbc.connect("DRIVER={SQL Server};server=IN2174234W2\MSSQLSERVER2014;database=P2P_RawData")
    conn = pyodbc.connect("DRIVER={SQL Server};server=" + ServerName + ";database=" + DBName + "")
    cursor = conn.cursor()
    cursor.execute(SourceQuery)
    for row in cursor:
        data.append(row.sentence_string)
        i += 1
    
    print("... Completed loading from DB")
    
    cursor.close()
    print("Calculating duplicates...(This may take sometime)")
    Language = "English"
    # Double-metaphone on each token
    l =[]
    l_result = []
    tokenized =[]
    
    #Russian language handling: translation
    if Language == 'Russian':
        for i in range(0,len(data)):
            tokenized.append((data[i].split(" "), i, data[i]))
            
            for n in range(0,len(tokenized[i][0])):
                token = translit(tokenized[i][0][n], 'ru', reversed=True)
                if token.isdigit(token) == True:
                    token = p.number_to_words(int(token)).strip()
                l.append((token,n,i, tokenized[i][2], doublemetaphone(token)[0],doublemetaphone(token)[1]))
    
    #Chinese language handling: translation
    elif Language == 'Chinese':
        pyin = Pinyin()
        for i in range(0,len(data)):
            tokenized.append((data[i].split(" "), i, data[i]))
            
            for n in range(0,len(tokenized[i][0])):
                token = pyin.get_pinyin(tokenized[i][0][n],'')
                if token.isdigit() == True:
                    tokenized[i][0][n] = p.number_to_words(int(token)).strip()
                l.append((token,n,i, tokenized[i][2], doublemetaphone(token)[0],doublemetaphone(token)[1]))
    
    elif Language == 'English':
        
        for i in range(0,len(data)):
            true_data.append(data[i])
            
            #Replace all special characters (non-alphanumeric) with space (" ") using reg-ex
            data[i] = re.sub('[^A-Za-z0-9]+', ' ',data[i])
            tokenized.append((data[i].split(" "), i, true_data[i]))
            
            for n in range(0,len(tokenized[i][0])):
                # Checking if the token is numeric: if numeric, we keep it as is (do not apply double-metaphone)
                if tokenized[i][0][n].isdigit() == True:
                    l.append((tokenized[i][0][n],n,i, tokenized[i][2], tokenized[i][0][n],tokenized[i][0][n]))
                else:
                    #double-metaphone calculations
                    dm_list = doublemetaphone(tokenized[i][0][n])
                    l.append((tokenized[i][0][n],n,i, tokenized[i][2], dm_list[0],dm_list[1]))
    
    # Scoring based on matched sounds
    df = pd.DataFrame(l,columns=['Word','Word_No', "Sentence_No", "Sentence", "Metaphone_Score1", "Metaphone_Score2"])
    df = df.groupby(['Sentence_No','Sentence']).agg(lambda x: tuple(x)).applymap(list).reset_index()
    if(len(df.index)==0):
        df["Score"] = 'None'
    else:
        df["Score"] = df["Metaphone_Score1"] + df["Metaphone_Score2"]


    r = df.values
    
    t=[]
	
    #calculate overlap percentage of the combined (2 lists of scores) double-metaphone
    for i1 in range(0,len(r)):
        for i2 in range(i1+1,len(r)):
            intersect_set_dm = set(r[i1][6]) & set(r[i2][6])
            score_dm = 200.0 * len(intersect_set_dm) / (len(set(r[i1][6])) + len(set(r[i2][6])))
			#Calculate jaro distance when double metaphone score exceeds the threshold 
            score_jaro = 0
            if score_dm >= Threshold_DM:
                score_jaro = d.jaro_distance(r[i1][1],r[i2][1])
            t.append((r[i1][0],r[i2][0],r[i1][1],r[i2][1],r[i1][6],r[i2][6],len(intersect_set_dm),score_dm,score_jaro )) 
    
    #Transformed t to a dataframe
    rst = pd.DataFrame.from_records(t, columns=["sn1", "sn2", "s1", "s2", "m1", "m2", "Overlap", "OverlapPercent", "JaroDist"])
	
    #Removal of transitive property (Greedy approach)
    idx = rst.groupby(['sn1'])['OverlapPercent'].transform(max) == rst['OverlapPercent']
    rst[idx][['sn1', 's1', 'sn2', 's2','m1','m2','OverlapPercent']].query('OverlapPercent >= ' + str(Threshold_DM) + '')
    l_result = rst[idx][['sn1', 's1', 'sn2', 's2','m1','m2','OverlapPercent', 'JaroDist']]
    
    #ranks based on highest overlap percentage. Row numbers for multiple match strings for one particular string
    l_result['rank'] = l_result['OverlapPercent'].rank(method='dense',ascending = 0).astype(int)
    l_result['row'] = l_result.sort_values(['rank','sn2','sn1'], ascending=[True,False,False]).groupby(['sn1','rank']).cumcount() + 1    
    
    #takes all rows with least rank; for every next rank we merge the records that are not already considered
    l_result[(l_result['rank'] == 1) & (l_result['row'] == 1)]
    x = l_result[(l_result['OverlapPercent'] >= Threshold_DM) & (l_result['row']==1)]
    ranks = list(set(x['rank']))
    
    for i in range(0,len(ranks)):
        if i==0 :
            s = x[(x['rank']==ranks[0])]
            p_temp = s
        else:
            xs = x[(x['rank'] == ranks[i])]
            cols_to_use = p_temp.columns.difference(s.columns)
            xs.set_index('sn2')
            p_temp.set_index('sn2')
            s = pd.merge(xs, p_temp[cols_to_use], left_index=True, right_index=True, how='outer',indicator=True)
            s = s[s['_merge']=='left_only']
            s = s[s['rank'].isnull() == 1]
            p_temp = pd.concat([p_temp,s], axis=0)
        p_temp = p_temp.reset_index(drop=True)
        
    p_final = p_temp[(p_temp['JaroDist'] >= Threshold_JD/100.0)]
    
    p_final = p_final[['sn1','s1', 'sn2', 's2', 'm1', 'm2', 'OverlapPercent', 'JaroDist' ]].values
    
    #Output: 'String#','String', 'CleanString#', 'CleanString', 'Metaphone of string', 'Metaphone of clean string', 'OverlapPercent'
    print("Loading to Destination...")
    cursor = conn.cursor()
    cursor.execute("delete from " + DestinationTableName + " where [Entity] = '"+EntityName+"' and [Property] = '"+PropertyName+"';")
    for i in range(0,len(p_final)):
        cursor.execute("insert into " + DestinationTableName + " ([Entity], [Property], [Sentence_no], [Sentence], [Sentence_Clean_no], [Sentence_Clean], [token_score], [token_clean_score], [Score], [JaroDistScore]) values ('"+EntityName+"','"+PropertyName+"',?,?,?,?,?,?,?,?)", p_final[i][0], p_final[i][1], p_final[i][2], p_final[i][3], ', '.join(map(str, p_final[i][4])), ', '.join(map(str, p_final[i][5])), p_final[i][6], p_final[i][7])
        cursor.commit()
    cursor.close()
    
    conn.close()
    
    end = time.time()
    print(end - start)

########################## END OF FUNCTION #####################################################

import sys
#INPUT from cmd: Server, DBName, Language
var1_server = sys.argv[1]
var2_dbproject = sys.argv[2]
var3_language = sys.argv[3]
print("Running - Vendor Name...")

#This is for Vendor - Name duplicate 
import pyodbc
conn = pyodbc.connect("DRIVER={SQL Server};server=" + var1_server + ";database=" + var2_dbproject + "")
conn.execute("exec [dbo].[usp_DM_save_inputlist_fuzzy] 'Vendor', 'Name', 5000 ")
conn.commit() 
save_fuzzy_duplicate(
        ServerName = var1_server,   
        DBName = var2_dbproject,  
        SourceQuery = "select [sentence_string] from [dbo].[fuzzy_input_temp]", 
        DestinationTableName = "[dbo].[FuzzyResults]", 
        EntityName = "Vendor", 
        PropertyName = "Name" ,
        Threshold_DM = 80,
        Threshold_JD = 75,
        Language = var3_language
        )
print("End of Vendor Name...")
print("Running - SP DUP Vendor...")
conn = pyodbc.connect("DRIVER={SQL Server};server=" + var1_server + ";database=" + var2_dbproject + "")
conn.execute("exec [dbo].[usp_load_report_duplicate_vendors]")
conn.commit() 

print("END - DUP Vendor...")