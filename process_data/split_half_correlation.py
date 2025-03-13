import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression
import random

# Load the human data
df = pd.read_csv('TurkData-Accuracy.csv')
columns_to_keep = ['subject', 'concept', 'list', 'response', 'set.number', 'right.answer']
df = df[columns_to_keep]

# Get unique subjects
subjects = df['subject'].unique()

# Function to calculate accuracy for a group of subjects
def calculate_group_accuracy(group_subjects, df):
    group_df = df[df['subject'].isin(group_subjects)]
    
    # Group by concept and list
    grouped = group_df.groupby(['concept', 'list'])
    
    results = {}
    
    for (concept, _list), group in grouped:
        key = f"CONCEPT_{concept}__LIST_{_list}"
        
        # Calculate accuracy at different stages for each subject
        subject_accuracies = []
        
        for subject, subject_group in group.groupby('subject'):
            response = (subject_group['response'] == 'T')
            right_ans = (subject_group['right.answer'] == 'T')
            sets = max(subject_group['set.number'])
            
            if sets < 25:
                continue
                
            correct_array = list(response == right_ans)
            outputs = correct_array
            
            if len(outputs) < 25:
                continue
                
            # Calculate accuracy at 25%, 50%, 75%, and 100%
            first_25 = outputs[: int(0.25 * len(outputs))]
            second_25 = outputs[int(0.25 * len(outputs)) : int(0.5 * len(outputs))]
            third_25 = outputs[int(0.5 * len(outputs)) : int(0.75 * len(outputs))]
            last_25 = outputs[-int(0.25 * len(outputs)) :]
            
            # Calculate accuracy for each quarter
            if len(first_25) > 0:
                first_acc = first_25.count(True)/len(first_25)
            else:
                first_acc = 0
                
            if len(second_25) > 0:
                second_acc = second_25.count(True)/len(second_25)
            else:
                second_acc = 0
                
            if len(third_25) > 0:
                third_acc = third_25.count(True)/len(third_25)
            else:
                third_acc = 0
                
            if len(last_25) > 0:
                last_acc = last_25.count(True)/len(last_25)
            else:
                last_acc = 0
                
            subject_accuracies.append([first_acc, second_acc, third_acc, last_acc])
        
        if subject_accuracies:
            # Average across subjects
            results[key] = np.mean(subject_accuracies, axis=0)
    
    return results

# Perform split-half correlation
n_iterations = 2
correlations = []

for i in range(n_iterations):
    # Randomly split subjects into two groups
    random.shuffle(subjects)
    split_point = len(subjects) // 2
    group1 = subjects[:split_point]
    group2 = subjects[split_point:]
    
    # Calculate accuracy for each group
    group1_results = calculate_group_accuracy(group1, df)
    group2_results = calculate_group_accuracy(group2, df)
    
    # Get common concepts between the two groups
    common_concepts = set(group1_results.keys()).intersection(set(group2_results.keys()))
    print(len(common_concepts))
    
    # Prepare data for correlation
    group1_data = []
    group2_data = []
    
    for concept in common_concepts:
        group1_data.extend(group1_results[concept])
        group2_data.extend(group2_results[concept])
    
    # Calculate correlation
    group1_data = np.array(group1_data).reshape(-1, 1)
    group2_data = np.array(group2_data)
    
    r = r_regression(group1_data, group2_data)[0]
    correlations.append(r)
    
    print(f"Iteration {i+1}/{n_iterations}: r({len(group2_data)-2}) = {r:.4f}")

# Calculate average correlation and standard deviation
avg_correlation = np.mean(correlations)
std_correlation = np.std(correlations)

print(f"\nAverage split-half correlation: r = {avg_correlation:.4f} ± {std_correlation:.4f}")
print(f"Spearman-Brown corrected reliability: {2*avg_correlation/(1+avg_correlation):.4f}") 