import pandas as pd

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df.to_csv("churn_data.csv", index=False)
print("✅ churn_data.csv downloaded and saved.")

