def floor_painting_bouns():
    """ 
    This is Python 2.7 code that defines how use 12 buckets of paint of 12 shades (4 paints X 3 shades in each paint) to cover the floor defined by a user with minimium waste.
    We have 12 paint buckets from 4 primary colors with each having 3 shades (i.e. total 12 buckets).
    The bucket size, or more specifically, the amount of area we can paint from each shade is given in the following table/matrix/arrays. 
    The different shades of the same primary color is shown on the same row. 
    
    [12, 23, 14]
    [10, 30, 15]
    [16, 22, 35]
    [14, 24, 20]
    
    Problem & Constraints
    We need to select 4 shades to paint the floor area such that;

    1. Entire floor area should be painted; also no overlaps of shades are allowed
    2. "One and only one" shade from each primary color has been selected for the final painting
    3. Amount of wastage is minimized (i.e. assume once we open and use a bucket, any remainings will be discarded)

    Inputs: 
    paint_footage = a matrix of paint shade (values in the row) with elements presenting paint per footage
    floor_size = floor size to be covered by paint

    Outputs with the data example:

    a) If floor_size > 0, then  produce following text to the screen:

    Selected shade combination: ['(0,0)', '(1,0)', '(2,2)', '(3,2)']
    Wastage: 2

    Note:
    If there are duplicates in total floor covered by paint, the function reports the first obsrevation.

    b) If floor_size <=0, then  produce following text to the screen:
        
    ERROR:
    No solution!
    You have entered invalid value for the floor size.

    c) If floor_size > max(paint_footage), then  produce following text to the screen:

    WARNING:
    You do not have enough paint to cover the whole floor.
    112 is the maximum footage that can be covered by the following combination of the shades: /n
    ('(0,1)', '(1,1)', '(2,2)', '(3,1)')
    888 is footage that is left unpainted
    """



    import pandas as pd
    import numpy as np
    import itertools

    print '\nThis solution looks for optimum paint shade allocation to cover the floor of the given size with minimum paint waste'

    floor_size =int(raw_input('Please, enter size of the floor to be painted\n > '))
    # Parameters of the paint matrix
    raws_no = 4 
    columns_no = 3
    
    def enter_data(dataframe, raws_no, columns_no):
        """ the function should enter the data into the paint_footage matrix"""
        for i in range(raws_no):
            for j in range(columns_no):
			print('Please enter (%d,%d) element of the paint footage matrix' %(i,j))
			value = float(input('Your value is: '))
			print(value)
			dataframe.set_value(i, j, value)
   
    def list_calc(in_dataframe, list_name):
        """ This function does date transformation into tuple """
        A0=np.array(in_dataframe.loc[0,:])
        B0=np.array(in_dataframe.loc[1,:])
        C0=np.array(in_dataframe.loc[2,:])
        D0=np.array(in_dataframe.loc[3,:])

            #list_name =list()
        for element in itertools.product(A0, B0, C0, D0):
            list_name.append(element)
            
    def ceil_key(d, key):
        """This function calculates the ceiling value that is closest to the required value"""
        if key in d:
            return key
        return min(k for k in d if k > key)
    
    
       # Creating the paint footage data
    data={}
    data =np.zeros((raws_no, columns_no))
    paint_footage=pd.DataFrame(data)
    
    enter_data(paint_footage,raws_no,columns_no)    
    
    # data = np.array([[12,23, 14],[10,30,15],[16,22,35],[14,24,20]])
    paint_footage=pd.DataFrame(data)
    e_list=list()
    list_calc(paint_footage, e_list)
    value_tuple= map(sum, e_list)
    
    # Creating coordinate data
    cdata = np.array([['(0,0)','(0,1)','(0,2)'],['(1,0)','(1,1)','(1,2)'],['(2,0)','(2,1)','(2,2)'],['(3,0)','(3,1)','(3,2)']])
    coord_matrix=pd.DataFrame(cdata)
    ce_list=list()
    list_calc(coord_matrix, ce_list)
        
    # Combining coordinate data with the paint footage
    paint_list = dict(zip(value_tuple,ce_list))
    max_value_tuple=max(value_tuple)


    max_value_tuple=max(value_tuple) 
    footage_left=floor_size  - max_value_tuple 
    # floor_size= 1000
    if floor_size <= 0:
        print '\x1b[6;33;95m' + 'ERROR:' + '\x1b[0m'
        print '\x1b[6;33;95m' + 'No solution! \nYou have entered invalid value for the floor size.' + '\x1b[0m'
    elif floor_size > max_value_tuple:
        print '\x1b[6;33;95m' + 'WARNING:' + '\x1b[0m'
        print 'You do not have enough paint to cover the whole floor.'
        print max_value_tuple, 'is the maximum footage that can be covered by the following combination of the shades:'
        print paint_list[max_value_tuple]
        print footage_left, 'is footage left unpainted'    
    else:
        optimum_key = ceil_key(paint_list, floor_size)

        # Test case:
        # floor_size=100
        # Out[]:101

        print"\nSelected optimum combination with minimum paint waste is:", paint_list[optimum_key]
        # paint_list[optimum_key].replace(''',' ')
        wastage = optimum_key - floor_size 
        print "Wastage is ", wastage

