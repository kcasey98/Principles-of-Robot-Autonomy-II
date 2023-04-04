import pandas as pd

print("hello")
# create a sample dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'San Francisco', 'Seattle', 'Los Angeles']
}
df = pd.DataFrame(data)

# export the dataframe to a CSV file
df.to_csv('data.csv', index=False)