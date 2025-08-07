import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
%matplotlib inline

df = pd.read_csv('train.csv')
df.head()

print("Shape of the dataset :",df.info())

df.isnull().sum()

df.tail()

df.describe()

df.shape

df.columns

df.nunique()
df['Age'].unique()

df['Pclass'].unique()

df['Survived'].unique()

df['Name'].unique()

#fill missing values with median 
df['Age'].fillna(df['Age'].median(), inplace=False)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=False)

df.drop('Cabin',axis=1, inplace=True)

df.drop(['Ticket','Name'],axis=1, inplace = True)

#Encode Sex and Embarked 
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1 , 'Q':2})

#Check final missing values 
df.isnull().sum()

df.groupby('Sex')['Survived'].mean()

df.groupby('Pclass')['Survived'].mean()

df.corr()

df.groupby('Sex')['Survived'].median()

df.groupby('Pclass')['Survived'].median()

#VISUALIZATION
df_numeric = df.apply(pd.to_numeric, errors='coerce')
df_numeric = df_numeric.dropna()
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Age Distribution")
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(df['Survived'], bins=10, kde=True)
plt.title("Survived Distribution")
plt.show()


    """Interactive CLI tool to plot histograms of numerical data with optional KDE curves."""
    while True:
        print("\nAvailable columns for histogram:")
        for i, col in enumerate(dataframe.columns, start=1):
            print(f"{i}. {col}")
        try:
            selection = int(input("\nChoose a column number to plot (or 0 to exit): "))
            if selection == 0:
                print("Exiting histogram visualization tool.")
                break
            selected_column = dataframe.columns[selection - 1]
            if not pd.api.types.is_numeric_dtype(dataframe[selected_column]):
                print(f"'{selected_column}' is not numeric. Please pick a numeric column.")
                continue
            bins_input = input("Number of bins? (default: 10): ").strip()
            bins = int(bins_input) if bins_input.isdigit() else 10
            kde_input = input("Add KDE overlay? (y/n, default: y): ").strip().lower()
            include_kde = False if kde_input == 'n' else True
            plt.figure(figsize=(15, 8))  
            sns.histplot(data=dataframe, x=selected_column, bins=bins, kde=include_kde)
            plt.title(f"Histogram of '{selected_column}'", fontsize=14)
            plt.xlabel(selected_column)
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except (ValueError, IndexError):
            print("Invalid input. Please select a valid column number.")
if __name__ == "__main__":
    try:
        print("Dataset 'train.csv' loaded successfully.")
        launch_histogram_tool(df)
    except Exception as e:
        print(f"Error loading file: {e}")




def plot_all_columns_as_categorical(df):
    """Interactive bar plot tool that treats every column as categorical."""
    while True:
        print("\nAvailable columns for plotting:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        try:
            choice = int(input("\nSelect a column number to plot (0 to exit): "))
            if choice == 0:
                print("Exiting the bar plot tool.")
                break
            selected_col = df.columns[choice - 1]
            value_counts = df[selected_col].astype(str).value_counts().head(20)  # Convert to string and take top 20
            plt.figure(figsize=(12, 7))
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Bar Plot of '{selected_col}'")
            plt.xlabel(selected_col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid column number.")
if __name__ == "__main__":
    try:
        print("Titanic dataset loaded successfully.")
        plot_all_columns_as_categorical(df)
    except Exception as e:
        print(f"Could not load the dataset: {e}")



plt.figure(figsize=(10,5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age and Survived')
plt.show()



plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age and Pclass')
plt.show()



plt.figure(figsize=(10,5))
sns.boxplot(x='Age', y='Embarked', data=df)
plt.title('Age and Embarked')
plt.show()


sns.pairplot(df)



g = sns.relplot(x='Survived', y='Age', hue='Sex', data=df)
plt.setp(g.ax.get_xticklabels(), rotation=60)
plt.show()




sns.distplot(df['Age'], bins=5)



sns.distplot(df['Survived'],bins=5)


sns.catplot(x='Age' , kind='box' , data=df)















