import matplotlib.pyplot as plt

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
    '''
    Making and showing plot of training points and linear function

    Args:
        x(list[flaot]): training data
        y(list[float]): target values
        w(scalar): model parameters
        b(scalar): model parameter
    '''
    plt.scatter(x,y)
    predictions = list()
    for i in range(len(x)):
        predictions.append(w*x[i]+b)
    plt.plot(x,predictions)
    plt.xlabel('Years of experience in 15')
    plt.ylabel('Years of experience in 125 000')
    plt.show()

def compute_cost(x:list[float],y:list[float],w:float,b:float):
    '''
    Computing squered error cost function

    Args:
        x(list[flaot]): training data
        y(list[float]): target values
        w(scalar): model parameters
        b(scalar): model parameter

    Returns:
        cost_sum(scalar): cost
    '''
    m = len(x)
    cost_sum = 0
    for i in range(m):
        y_p = w*x[i]+b
        cost_sum += (y_p-y[i])**2
    cost_sum /= 2*m
    return cost_sum

def compute_gradient(x:list[float],y:list[float],w:float,b:float):
    '''
    Computing derivatives for gradient descent

    Args:
        x(list[flaot]): training data
        y(list[float]): target values
        w(scalar): model parameters
        b(scalar): model parameter

    Returns:
        dj_dw(scalar): value of derivative
        dj_db(scalar): value of derivative
    '''
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
    '''
    Computing gradient descent

    Args:
        x(list[flaot]): training data
        y(list[float]): target values
        w(scalar): model parameters
        b(scalar): model parameter
        alpha(scalar): learning rate
        it(scalar): number of iterations

    Returns:
        w_g , b_g(scalars): new values for parameteres w an b
    '''
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
    '''
    Printing predicted value for any x
    
    Args: 
        x(scalar): any number 
        w(scalar): model parameter
        b(scalar): model parameter
    '''
    y = w*x+b
    print('Predicted value of salary for someone who works for ',x,' years is: ',y*125000)
