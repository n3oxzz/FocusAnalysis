import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("focus_score.csv")

focus_scores = df["Focus Score"].tolist()

length_of_scores = len(focus_scores)
third = length_of_scores // 3

part1 = focus_scores[:third]
part2 = focus_scores[third:third*2]
part3 = focus_scores[third*2:]

#mean 
def calculate_mean(part):
    # Calculates the arithmetic mean (average) of a data segment (part)
    # part: a list of focus scores (e.g., part1, part2, part3)
    # len(part): total number of data points in this segment
    # sum(part): total of all focus scores in this segment
    return sum(part) / len(part)

mean1 = calculate_mean(part1)
mean2 = calculate_mean(part2)
mean3 = calculate_mean(part2)

print(f"First mean: {mean1}\nSecond mean: {mean2}\nThird mean:{mean3}")

#standard deviation
def calculate_std(part, mean):
    # Calculates the standard deviation of a data segment
    # part: list of focus scores
    # mean: mean value of this part, used to calculate squared differences
    # For each value in the part, we compute (val - mean)^2, take the average, and then square root
    return (sum((val - mean) ** 2 for val in part) / len(part)) ** 0.5

SD1 = calculate_std(part1, mean1)
SD2 = calculate_std(part2, mean2)
SD3 = calculate_std(part3, mean3)

print(f"SD1: {SD1}\nSD2: {SD2}\nSD3: {SD3}")

#moving avg
def moving_avg(data, window):
    # Calculates the moving average over a list of data using a specified window size.

    # Parameters:
    # - data: list of numerical values (e.g., focus scores over time)
    # - window: integer, the number of points to average over

    # Returns:
    # - A list where each value is the average of the current point and the previous (window - 1) points.(For the first few elements, it averages as many values as are available.
    return [sum(data[max(0, i - window + 1): i + 1]) / (i - max(0, i - window + 1) + 1)for i in range(len(data))]

smoothed_scores = moving_avg(focus_scores, window=5)

#graphs of every value
plt.figure(figsize=(12, 6))
plt.plot(range(len(focus_scores)), focus_scores, label="Focus Score (full)", color="purple")
plt.xlabel("Time (s)")
plt.ylabel("Focus Score")
plt.title("Focus Score Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#graph separated into 3 equal periods
plt.figure(figsize=(12, 6))
plt.plot(range(len(part1)), part1, label=f"Part 1 (Mean={mean1:.2f}, SD={SD1:.2f})", color="blue")
plt.plot(range(third, third + len(part2)), part2, label=f"Part 2 (Mean={mean2:.2f}, SD={SD2:.2f})", color="green")
plt.plot(range(third*2, third*2 + len(part3)), part3, label=f"Part 3 (Mean={mean3:.2f}, SD={SD3:.2f})", color="red")
plt.xlabel("Time(s)")
plt.ylabel("Focus Score")
plt.title("Focus Score Over Time (3 parts) ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#moving avg scores graph
plt.figure(figsize=(10, 4))
plt.plot(range(len(focus_scores)), smoothed_scores, label="Smooth Focus Score", color='darkorange')
plt.xlabel("Time(s)")
plt.ylabel("Focus Score")
plt.title("Focus Score Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()











