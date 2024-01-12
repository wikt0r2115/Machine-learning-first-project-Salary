def load_data(name:str):
    '''
    Loading the data from txt file

    Args:
        name(scalar):file name
    
    Returns:
        x(list[flaot]): list of x_training values
        y(list[float]): list of y_training values
    '''
    x = list()
    y = list()
    with open(name) as file:
        for line in file:
            data_set = line.split()
            x.append(float(data_set[1])/15)
            y.append(float(data_set[2])/125000)
    return x,y

def training_data_visualization(x:list,y:list,w:float,b:float):
    plt.scatter(x,y)
    predictions = list()
    for i in range(len(x)):
        predictions.append(w*x[i]+b)
    plt.plot(x,predictions)
    plt.xlabel('Years of experience in 15')
    plt.ylabel('Years of experience in 125 000')
    plt.show()

def compute_cost(x:list[float],y:list[float],w:float,b:float):
    m = len(x)
    cost_sum = 0
    for i in range(m):
        y_p = w*x[i]+b
        cost_sum += (y_p-y[i])**2
    cost_sum /= 2*m
    return cost_sum
'''
def cost_3D_visualization(x:list[float],y:list[float]):
    w_array = list()
    for i in range(0,400+1,5):
        w_array.append(i)
    b_array = w_array
    #print(w_array)
    m = len(w_array)
    cost = list()
    for i in range(m):
        cost.append(compute_cost(x,y,w_array[i],b_array[i]))
    ax = plt.axes(projection='3d')
    ax.scatter(w_array,b_array,cost)
    plt.show()
'''
def compute_gradient(x:list[float],y:list[float],w:float,b:float):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i]+b
        dj_dw += (f_wb-y[i])*x[i]
        dj_db += f_wb-y[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw,dj_db

def gradient_descent(x:list[float],y:list[float],w:float,b:float,alpha:float,it:float):
    w_g = w
    b_g = b
    for i in range(int(it)):
        dj_dw , dj_db = compute_gradient(x,y,w_g,b_g)
        w_temp = w_g - alpha * dj_dw
        b_temp = b_g - alpha * dj_db
        if i%10e3 == 0:
            cost = compute_cost(x,y,w_temp,b_temp)
            print(i,' cost: ',cost)
        w_g = w_temp
        b_g = b_temp
    return w_g , b_g

def prediction(x:float,w:float,b:float):
    y = w*x+b
    print('Predicted value of salary for someone who works for ',x,' years is: ',y*125000)
#biblioteka do wizualizacji danych
import matplotlib.pyplot as plt
import numpy as np
#lista x przechowuje dane wejściowe 
x_train = list()
#lista y przechowuje dobre odpowiedzi do danych treningowych 
y_train = list()

x_train , y_train = load_data('Salary_dataset.txt')
w_init = 1.133995478574576
b_init = 0.19878563173219757
alpha = 0.1
iterations = 10e+5

#print(x_train[-1],y_train[-1])
#print(x_train[0:3], type(x_train[0]))
#print(y_train[0:3], type(y_train[0]))

training_data_visualization(x_train,y_train,w_init,b_init)

#cost_3D_visualization(x_train,y_train)
#print(compute_cost(x_train,y_train,w_init,b_init))

#w , b = gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations)
#print('w,b after ',iterations,' w:',w,' b:',b)

prediction(2.1/15,w_init,b_init)
#ZMIENIONA WERSJA