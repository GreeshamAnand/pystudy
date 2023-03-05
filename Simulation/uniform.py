import numpy as np

def simulate_overlap():
    task1 = np.random.randint(0, 300, size = 100000) #outputs a 1000 size array
    task2 = np.random.randint(0, 300, size = 100000)
    
    overlap_percentage = np.mean(np.abs(task1 - task2) <= 60)
    overlap_median = np.median(np.abs(task1 - task2) <= 60)

    return overlap_percentage, overlap_median

if __name__ == "__main__":
    a, b = simulate_overlap() 
    print(a*1000*365)
    print("Median Cost {}" .format(b*1000*365))