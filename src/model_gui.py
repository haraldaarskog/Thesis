

# Importing the matplotlb.pyplot
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import model_functions as mf



def transform_dict(d):
    queue_set=set()
    some_d={}
    for key in d:
        queue=key[0]
        value=d[key]
        if queue not in queue_set:
            queue_set.add(queue)
            some_d[queue]=[]
        else:
            if value>0:
                some_d[queue].append(key[1])
    arr1=[]
    for k in some_d:
        arr2=[]
        for value in some_d[k]:
            t=(value, 1)
            arr2.append(t)
        arr1.append(arr2)
    return arr1

def create_gantt_chart(Queues, Time_periods, variable_dict):
    # Declaring a figure "gnt"
    #fig, gnt = plt.subplots()
    x_size=20
    y_size=8
    fig = plt.figure(figsize=(x_size,y_size))
    ax = fig.add_subplot(111)

    # Setting labels for x-axis and y-axis
    ax.set_xlabel('Time periods')
    ax.set_ylabel('Queues')

    # Setting Y-axis limits
    ax.set_ylim(0, Queues*10+2)

    # Setting X-axis limits
    ax.set_xlim(0, Time_periods)

    # Setting ticks on y-axis
    ax.set_yticks(np.arange(10, Queues*10+1, 10))

    # Labelling tickes of y-axis
    ax.set_yticklabels(np.arange(1, Queues, 1).astype(str))

    ax.set_xticks(np.arange(0, Time_periods, 1))
    ax.set_yticklabels(np.arange(0, Time_periods, 1).astype(str))

    # Setting graph attribute
    ax.grid(True)
    ax.grid(color = 'g', linestyle = '-')
    #font = font_manager.FontProperties(size='small')
    #ax.legend(loc=1,prop=font)
    #ax.invert_yaxis()
    arr=[]
    queue_set={}
    queue_dict={}
    number=8
    arrays=transform_dict(variable_dict)
    for array in arrays:
        #ax.text(0.5, c, 'matplotlib', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        color = list(np.random.rand(3))
        ax.broken_barh(array, (number, 4), facecolor=color)
        number+=10



    plt.savefig("gantt1.png")

b_jt=mf.loadSolution("output/model_solution.sol")["b"]
#create_gantt_chart(3, 14, b_jt)
