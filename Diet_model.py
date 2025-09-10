import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn.utils import resample 
import matplotlib.pyplot as plt 
 
# Load the food and nutrient data 
print("Loading and processing food data...") 
 
# Assuming 'food.csv' and 'nutrients.csv' are in the same directory as this script 
foods = pd.read_csv('foods.csv') 
nutrients = pd.read_csv('nutrients.csv') 
 
# Merge the two dataframes based on the 'ID' column 
df = pd.merge(foods, nutrients, on='ID') 
 
# Check the first few rows to understand the merged data 
print(df.head()) 
 
# 👉 Analyze the nutritional value distributions 
print("\nNutrient stats:\n", df[['TotalFat', 'Carbohydrate', 'Protein']].describe()) 
 
# Define the target variable (diet type) based on food features 
def get_diet_type(row): 
    fat = row['TotalFat'] 
    carbs = row['Carbohydrate'] 
    protein = row['Protein'] 
 
    # Diet classification based on features 
    if fat > 20 and carbs < 15: 
        return 'Keto' 
    elif protein > 25 and fat < 10: 
        return 'High-Protein Low-Fat' 
    elif carbs < 20 and fat < 15: 
        return 'Low-Carb' 
    elif fat < 8 and carbs > 35 and protein < 15: 
        return 'Vegan' 
elif 10 <= fat <= 20 and 25 <= carbs <= 45 and 10 <= protein <= 25: 
        return 'Mediterranean' 
    else: 
        return 'Balanced' 
 
# Apply the function to create a 'diet_type' column 
df['diet_type'] = df.apply(get_diet_type, axis=1) 
 
# Check class distribution before balancing 
print("\nDiet type distribution (before balancing):\n", df['diet_type'].value_counts()) 
 
# Step 2: Balance the dataset by random undersampling 
grouped = [resample(group, 
                    replace=False, 
                    n_samples=min(df['diet_type'].value_counts().values), 
                    random_state=42) 
           for _, group in df.groupby('diet_type')] 
 
# Combine the balanced groups 
df_balanced = pd.concat(grouped) 
 
# Shuffle the dataframe 
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) 
 
# Check class distribution after balancing 
print("\nDiet type distribution (after balancing):\n", df_balanced['diet_type'].value_counts()) 
 
# Step 3: Visualize class distribution before and after balancing 
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
 
df['diet_type'].value_counts().plot(kind='bar', ax=axes[0], color='lightblue') 

axes[0].set_title('Before Balancing') 
axes[0].set_xlabel('Diet Type') 
axes[0].set_ylabel('Count') 
 
df_balanced['diet_type'].value_counts().plot(kind='bar', ax=axes[1], color='lightcoral') 
axes[1].set_title('After Balancing') 
axes[1].set_xlabel('Diet Type') 
axes[1].set_ylabel('Count') 
 
plt.tight_layout() 
plt.show() 
 
# Features and target 
# Add synthetic user features for training purposes 
df_balanced['age'] = np.random.randint(18, 65, df_balanced.shape[0]) 
df_balanced['bmi'] = np.random.uniform(18.5, 30.0, df_balanced.shape[0]) 
df_balanced['activity_level'] = np.random.randint(1, 6, df_balanced.shape[0]) 
df_balanced['vegetarian'] = np.random.randint(0, 2, df_balanced.shape[0]) 
 
# Now use those columns for training 
X = df_balanced[['age', 'bmi', 'activity_level', 'vegetarian']] 
y = df_balanced['diet_type'] 
 
 
# Split data into training and testing sets (80% train, 20% test) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Initialize and train the Random Forest model 
model = RandomForestClassifier() 
model.fit(X, y) 
 
# Evaluate the model on the test set 
y_pred = model.predict(X_test) 
# Calculate accuracy 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy * 100:.2f}%") 
 
 
# Print the classification report 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
# Print the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("\nConfusion Matrix:") 
print(conf_matrix) 
# Optionally, save the trained model for future use 
import joblib 
joblib.dump(model, 'diet_recommender_rf.pkl') 
 
print("\nModel has been saved as 'diet_recommender_rf.pkl'")
