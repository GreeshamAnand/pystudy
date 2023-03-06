import numpy as np
import random

def simulate_overlap():
    task1 = np.random.randint(0, 300, size = 100000) #outputs a 1000 size array
    task2 = np.random.randint(0, 300, size = 100000)
    
    overlaps = np.abs(task1 - task2) <= 60

    overlap_percentage = np.mean(overlaps)
    overlap_median = np.median(overlaps)

    return overlap_percentage, overlap_median


def rand_starttime():
    return random.uniform(7,12)


def simulate_crash_cost():

    # Create a random uniform startime for the two jobs
    

    overlap = 0 
    for _ in range(0,365):

        tt1 = rand_starttime()
        tt2 = rand_starttime()

        if abs(tt1 - tt2) < 1:
            overlap += 1

    return overlap*1000




if __name__ == "__main__":
    a, b = simulate_overlap() 
    print(a*1000*365)
    print("Median Cost {}" .format(b*1000*365))

    ## Simulate the yearly cost over 10000 times

    median_cost =  np.median([simulate_crash_cost() for i in range(10000)])
    print(median_cost)
    mean_cost = np.mean([simulate_crash_cost() for i in range(10000)])
    print(mean_cost)