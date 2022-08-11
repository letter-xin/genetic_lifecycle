import numpy as np
import math
import time
import pandas as pd
from tqdm import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
path_df = pd.read_csv('storeData.csv')

path_df = pd.read_csv('storeData.csv').drop(path_df.columns[[0]], axis=1)
path_df
messageDF=pd.read_csv('messageStore.csv')
messageDFValues=messageDF.loc[:,['better_distribution']]
messageDFValues['index']=np.arange(len(messageDFValues))
messageDFValues=messageDFValues[['index','better_distribution']]
messageDFValues['better_distribution']=messageDFValues.better_distribution*0.66
# messageDFValues.values
# selectFrame=pd.read_csv('messageStore.csv')




def get_route_fitness_value(route, dist_matrix,messageData):
    """
    :param route: ndarray
    :param dist_matrix:  ndarray
    :return:  double
    """
    # dist_sum = 0
    # for i in range(len(route)-1):
    #     dist_sum += dist_matrix[route[i], route[i+1]]
    # dist_sum += dist_matrix[route[len(route)-1], route[0]]


    s = 0
    startPoint=1.2667
    carry=0
    return_num=0
    DistanceNumber=0
    for i in range(route.shape[0]-1): #shape[0]
        #s += math.sqrt(np.sum(np.power(cities[i+1,:]-cities[i,:],2))) #np.power(x,y)
        
        carry+= messageData[route[i]]

        if carry>5000:
            s+=dist_matrix[int(route[i])][0]*carry*1.8513+startPoint*dist_matrix[0][int(route[i+1])]
            return_num+=1
            DistanceNumber+=dist_matrix[int(route[i])][0]+dist_matrix[0][int(route[i+1])]
            carry=0
        else :
            # print(path_values[values[i][0]][values[i][0]+1])
            #print(values[i][0]+1)
            s+= dist_matrix [int(route[i])] [int(route[i+1])]  *carry*1.8513
            DistanceNumber+=dist_matrix [int(route[i])] [int(route[i+1])]
    return (1/s,return_num,DistanceNumber)

def get_all_routes_fitness_value(routes, dist_matrix,messageData):
    """
    :param routes:  ndarray
    :param dist_matrix:  ndarray
    :return:  ndarray
    """
    fitness_values = np.zeros(len(routes))
    for i in range(len(routes)):
        f_value = get_route_fitness_value(routes[i], dist_matrix,messageData)[0]
        fitness_values[i] = f_value
    return fitness_values

def init_route(n_route, n_cities):
    """
    :param n_route:  int
    :param n_cities:  int
    """
    routes = np.zeros((n_route, n_cities)).astype(int)
    # print(type(aa))
    # print(n_route)
    aa_copy=np.array(aa)
    for i in range(n_route):
        # if i==0:
        #     routes[0]=np.array(aa)
        #     continue
        # np.random.shuffle((aa_copy))
        routes[i] = aa_copy
    return routes

def selection(routes, fitness_values):
    """
    :param routes:  ndarray
    :param fitness_values:  ndarray
    :return:  ndarray
    """
    selected_routes = np.zeros(routes.shape).astype(int)
    probability = fitness_values / np.sum(fitness_values)
    n_routes = routes.shape[0]
    for i in range(n_routes):
        choice = np.random.choice(range(n_routes), p=probability)
        selected_routes[i] = routes[choice]
    return selected_routes

def crossover(routes, n_cities):
    """
    :param routes:  ndarray
    :param n_cities:  int
    :return:  ndarray
    """
    for i in range(0, len(routes), 2):
        r1_new, r2_new = np.zeros(n_cities), np.zeros(n_cities)
        seg_point = np.random.randint(0, n_cities)
        cross_len = n_cities - seg_point
        r1, r2 = routes[i], routes[i+1]
        r1_cross, r2_cross = r2[seg_point:], r1[seg_point:]
        r1_non_cross = r1[np.in1d(r1, r1_cross)==False]
        r2_non_cross = r2[np.in1d(r2, r2_cross)==False]
        r1_new[:cross_len], r2_new[:cross_len] = r1_cross, r2_cross
        r1_new[cross_len:], r2_new[cross_len:] = r1_non_cross, r2_non_cross
        routes[i], routes[i+1] = r1_new, r2_new
    return routes

def mutation(routes, n_cities):
    """
    :param routes:  ndarray
    :param n_cities:  int
    :return:  ndarray
    """
    prob = 0.1
    p_rand = np.random.rand(len(routes))
    for i in range(len(routes)):
        if p_rand[i] < prob:
            mut_position = np.random.choice(range(n_cities), size=2, replace=False)
            l, r = mut_position[0], mut_position[1]
            routes[i, l], routes[i, r] = routes[i, r], routes[i, l]
    return routes

def cycleCaculate(para):
    start = time.time()

    n_routes = 150 
    epoch = 500 
    #MessageMYData
    tempn=len(messageDFValues)
    MessageMYData=messageDFValues.values[:,1][:tempn]
    cities = messageDFValues.values[:,0][:tempn]
    dist_matrix =  path_df.values[:tempn,:tempn]
    routes = init_route(n_routes, dist_matrix.shape[0]) 
    fitness_values = get_all_routes_fitness_value(routes, dist_matrix,MessageMYData) 
    best_index = fitness_values.argmax()
    best_route, best_fitness = routes[best_index], fitness_values[best_index] 
    not_improve_time = 0
    for i in tqdm(range(epoch)):
        routes = selection(routes, fitness_values) 
        routes = crossover(routes, len(cities)) 
        routes = mutation(routes, len(cities)) 
        fitness_values = get_all_routes_fitness_value(routes, dist_matrix,MessageMYData)
        best_route_index = fitness_values.argmax()
        if fitness_values[best_route_index] > best_fitness:
            not_improve_time = 0
            best_route, best_fitness = routes[best_route_index], fitness_values[best_route_index] 
        else:
            not_improve_time += 1

    #end = time.time()
    finalValue=get_route_fitness_value(best_route, dist_matrix,MessageMYData)[0]
    returnNum=get_route_fitness_value(best_route, dist_matrix,MessageMYData)[1]
    distance=get_route_fitness_value(best_route, dist_matrix,MessageMYData)[2]
    #print('time: {}s'.format(end-start))
    tempDf=pd.DataFrame({'best_route':str(best_route),'best_value':1/finalValue,'returnNum':returnNum,'Distance':distance,'best_routeRes':[list(best_route)]},index=[para])
    return tempDf

def plotCostPicture(x,y):
    # x = np.arange(1, len(ggList)+1)
    # y = ggList
    # z = [37, 25, 17, 49, 27, 77, 34, 34, 34, 51, 39, 52, 47, 12]
    # u = [37, 31, 19, 57, 29, 86, 36, 37, 45, 64, 42, 57, 50, 24]
    
    
    plt.style.use("default")
    plt.figure(figsize=(20,8),dpi=150)  
    plt.bar(x=x, height=y, label='AllCost', color='Coral', alpha=0.9)
    # plt.bar(x=x, height=u, label='sum', color='LemonChiffon', alpha=0.8)
    plt.legend(loc="upper left")

    plt.title("Detection results")
    plt.xlabel("experiment cost")
    plt.ylabel("Cost")
    
    

    # ax2.set_ylabel("recall")
    # ax2.set_ylim([0.5, 1.05]);
    plt.plot(x, y, "r", marker='.', c='r')
    for a, b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
    plt.legend(loc="upper right")
    plt.savefig('costPicture.jpg')

    
if __name__ == '__main__':
    
    range_number=50
    save_best_value=[]
    bigEpoch=3
    finalDf=pd.DataFrame()
    aa = [129, 40, 48, 45, 76, 59, 62, 30, 49, 54, 52, 65, 58, 
    50, 111, 57, 56, 64, 63, 53, 60, 51, 68, 69, 70, 71, 72, 132, 
    108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 8, 120, 
    121, 6, 7, 9, 161, 91, 87, 92, 86, 10, 158, 156, 157, 155, 176, 
    163, 162, 164, 165, 166, 169, 168, 170, 167, 43, 66, 179, 173, 
    14, 34, 15, 47, 46, 41, 61, 44, 55, 42, 36, 37, 17, 18, 19, 20, 
    21, 22, 23, 24, 84, 26, 27, 28, 16, 172, 178, 177, 192, 180, 189, 
    184, 174, 73, 74, 75, 77, 78, 82, 79, 88, 80, 81, 0, 135, 145, 146, 
    147, 188, 149, 150, 151, 154, 152, 153, 89, 185, 186, 191, 148, 183, 
    182, 181, 193, 194, 175, 171, 195, 196, 197, 198, 199, 200, 201, 25, 
    1, 2, 3, 4, 5, 39, 13, 85, 83, 90, 98, 99, 96, 93, 95, 100, 94, 97, 
    67, 107, 127, 190, 187, 11, 126, 128, 130, 140, 138, 137, 133, 139, 
    159, 160, 142, 141, 144, 136, 134, 143, 101, 102, 104, 105, 106, 103, 
    31, 32, 33, 29, 12, 38, 123, 124, 35, 125, 131, 122]
    for number in range(bigEpoch):
        myDf=pd.DataFrame()
        for i in range(range_number):
            
            tempDf=cycleCaculate(i)
            # aa=tempDf.best_routeRes.values[0]
            save_best_value.append(tempDf['best_value'].values[0])
            myDf=pd.concat([myDf,tempDf])
        # x=np.arange(1, len(save_best_value)+1)
        # y=save_best_value
        # plotCostPicture(x,y)
        aa=myDf.iloc[myDf.best_value.argmin()].best_routeRes   
        finalDf=pd.concat([finalDf,myDf],axis=0)
    finalDf.to_csv('finalDf.csv')