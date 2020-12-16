import pandas as pd

''' Split a string input by '||' and append
 the string tokens  into 
columns'''

string_input = input(str())

string_input = string_input.replace("||", ",")

data = ['Identifier, CountryCode, DeliveryMethod, Segment, Localproduct1, LocalProduct2, GlobalProduct']
data.append(string_input)
data = pd.DataFrame(data, columns=['row'])

frame= pd.DataFrame(data.row.str.split(',',6).tolist(), columns = ['Identifier','CountryCode', 'DeliveryMethod', 'Segment', 'Localproduct1', 
                                                                   'LocalProduct2', 'GlobalProduct'])
frame.drop(index=0,inplace=True)

GlobalProduct =  frame['LocalProduct2']
frame['GlobalProduct'] = GlobalProduct

localproduct1 = frame.Localproduct1.str[:11]
localproduct1 = localproduct1.str.replace('{', '')
localproduct1 = localproduct1.str.replace('}', '')
localproduct2 =  frame.Localproduct1.str[14:]
localproduct2 = localproduct2.str.replace('[', '')
localproduct2 = localproduct2.str.replace(']', '')
frame['Localproduct1'] = localproduct1
frame['LocalProduct2'] = localproduct2

#cleaning 
frame = frame.drop('Identifier', axis = 1)
frame