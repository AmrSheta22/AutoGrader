# Try Error Aalysis frequencies of the different types of substitutions
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np

def calculate_wer(original_texts, predicted_texts):
    substitution_counts = {}

    for original_text, predicted_text in zip(original_texts, predicted_texts):
        original_text = str(original_text)
        predicted_text = str(predicted_text)

        ref_words = original_text.split()
        hyp_words = predicted_text.split()

        substitutions = []

        for ref, hyp in zip(ref_words, hyp_words):
            if ref != hyp:
                # Substitutions
                diff_sequence_ref = ''.join(ref_char if ref_char.lower() != hyp_char.lower() else '' for ref_char, hyp_char in zip(ref, hyp))
                diff_sequence_hyp = ''.join(hyp_char if ref_char.lower() != hyp_char.lower() else '' for ref_char, hyp_char in zip(ref, hyp))
                # substitution two letters or less only
                if len(diff_sequence_ref) > 2:
                    pass
                else:
                    substitutions.append((diff_sequence_ref, diff_sequence_hyp))
        
        # Remove empty substitutions
        substitutions = [(diff_sequence_ref, diff_sequence_hyp) for diff_sequence_ref, diff_sequence_hyp in substitutions if diff_sequence_ref != '' and diff_sequence_hyp != '']
        
        # Count the occurrences of different types of substitutions
        for substitution in substitutions:
            diff_sequence_ref, diff_sequence_hyp = substitution
            substitution_key = (diff_sequence_ref, diff_sequence_hyp)
            substitution_counts[substitution_key] = substitution_counts.get(substitution_key, 0) + 1

    return substitution_counts


wer_substitution_counts = calculate_wer(original_texts, predicted_texts)

# Print the types of substitutions and their counts
for substitution, count in wer_substitution_counts.items():
    diff_sequence_ref, diff_sequence_hyp = substitution
    print(f"Substitution: {diff_sequence_ref} -> {diff_sequence_hyp}, Count: {count}")


def analyze_substitutions(substitution_counts):
    # Get the total count of substitutions
    total_substitutions = sum(substitution_counts.values())

    # Calculate the percentage of each substitution type
    substitution_percentages = {
        substitution: (count / total_substitutions) * 100
        for substitution, count in substitution_counts.items()
    }

    # Sort the substitution types by count in descending order
    sorted_substitutions = sorted(
        substitution_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Print the substitution types and their percentages
    print("Substitution Analysis:")
    for substitution, count in sorted_substitutions:
        diff_sequence_ref, diff_sequence_hyp = substitution
        percentage = substitution_percentages[substitution]
        print(f"Substitution: {diff_sequence_ref} -> {diff_sequence_hyp}")
        print(f"Count: {count}")
        print(f"Percentage: {percentage:.2f}%")
        print()

    if sorted_substitutions:
        # Identify the most common substitution type
        most_common_substitution = sorted_substitutions[0][0]
        print("Most Common Substitution:")
        print(f"Substitution: {most_common_substitution[0]} -> {most_common_substitution[1]}")
        print(f"Count: {substitution_counts[most_common_substitution]}")
        print(f"Percentage: {substitution_percentages[most_common_substitution]:.2f}%")
    else:
        print("No substitutions found.")

# Call the function with the substitution counts
analyze_substitutions(wer_substitution_counts)

def analyze_substitutions(substitution_counts):
    # Get the total count of substitutions
    total_substitutions = sum(substitution_counts.values())

    # Calculate the percentage of each substitution type
    substitution_percentages = {
        substitution: count / total_substitutions * 100
        for substitution, count in substitution_counts.items()
    }

    # Sort the substitution types by count in descending order
    sorted_substitutions = sorted(
        substitution_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Identify the most common substitution type
    most_common_substitution = sorted_substitutions[0][0]
    print("Most Common Substitution:")
    print(f"Substitution: {most_common_substitution[0]} -> {most_common_substitution[1]}")
    print(f"Count: {substitution_counts[most_common_substitution]}")
    print(f"Percentage: {substitution_percentages[most_common_substitution]:.2f}%")

    # Group similar substitution types
    grouped_substitutions = {}
    for substitution, count in substitution_counts.items():
        diff_sequence_ref, diff_sequence_hyp = substitution
        group_key = diff_sequence_ref.lower()  # Group by lowercase reference
        if group_key not in grouped_substitutions:
            grouped_substitutions[group_key] = 0
        grouped_substitutions[group_key] += count

    # Sort the grouped substitution types by count in descending order
    sorted_grouped_substitutions = sorted(
        grouped_substitutions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Print the grouped substitution types and their counts
    print("\nGrouped Substitution Analysis:")
    for group, count in sorted_grouped_substitutions:
        print(f"Group: {group}")
        print(f"Count: {count}")
        print()

    # Compare with predefined expected substitutions
    expected_substitutions = {
        ("l", "1"),
        ("o", "0"),
        ("i", "1"),
        ("v", "r"),
        ("a","u"),
        ("a","e"),        
        ("e","a"),        
        ("r","n"),
        ("ll","dd"),
        ("a","o"),
        ("w'","ul"),
        ("n","m"),
        ("h","b"),
        ("r","p"),
        ("rl","nd"),
        ("de","ob"),
        ("t","l"),
        ("t","f"),
        ("f","r"),
        ("e","s"),
        ("or","an"),
        ("re","sa"),
        ("lv","br"),
        ("w","u"),
        ("w","v"),
        ("v","u"),
        ("n","v"),
        ("h","n"),
        ("h","l"),
        ("l","b"),
        ("l","I"),
        ("n","u"),
        ("o" , "a"),
        ("n","r"),
        # Add more expected substitutions as needed
    }

    # Identify unexpected substitutions
    unexpected_substitutions = set(substitution_counts.keys()) - expected_substitutions

    # Sort the unexpected substitutions by count in descending order
    sorted_unexpected_substitutions = sorted(
        unexpected_substitutions,
        key=lambda x: substitution_counts[x],
        reverse=True
    )

    print("Unexpected Substitutions:")
    for substitution in sorted_unexpected_substitutions:
        diff_sequence_ref, diff_sequence_hyp = substitution
        count = substitution_counts[substitution]
        percentage = substitution_percentages[substitution]
        print(f"Substitution: {diff_sequence_ref} -> {diff_sequence_hyp}")
        print(f"Count: {count}")
        print(f"Percentage: {percentage:.2f}%")
        print()
        
# Call the function with the substitution counts
analyze_substitutions(wer_substitution_counts)


def visualize_substitutions(substitution_counts, group_size=10):
    # Sort the substitution_counts dictionary by counts in descending order
    sorted_substitutions = sorted(
        substitution_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Extract substitution types and counts from the sorted substitutions
    substitutions = [substitution[0] for substitution in sorted_substitutions]
    counts = [substitution[1] for substitution in sorted_substitutions]

    # Split substitution types into groups
    grouped_substitutions = [substitutions[i:i+group_size] for i in range(0, len(substitutions), group_size)]
    grouped_counts = [counts[i:i+group_size] for i in range(0, len(counts), group_size)]

    # Create a bar chart for each group
    fig, axs = plt.subplots(len(grouped_substitutions), figsize=(12, 6 * len(grouped_substitutions)))

    # Customize the chart for each group
    for i, (substitutions_group, counts_group) in enumerate(zip(grouped_substitutions, grouped_counts)):
        ax = axs[i]
        ax.bar(range(len(substitutions_group)), counts_group)

        ax.set_xlabel('Substitution Types')
        ax.set_ylabel('Count')
        ax.set_title(f'Substitution Analysis (Group {i+1})')

        substitution_labels = [' -> '.join(substitution) for substitution in substitutions_group]
        ax.set_xticks(range(len(substitutions_group)))
        ax.set_xticklabels(substitution_labels, rotation='vertical', fontsize=8, ha='center')

    # Display the chart
    plt.tight_layout()
    plt.show()

# Call the function with the substitution counts
visualize_substitutions(wer_substitution_counts, group_size=50)


#observe the hierarchical relationships and similarities between the groups of substitution types.
#Clusters that are closer to each other on the dendrogram are more similar, while clusters that are farther apart are more dissimilar.
def visualize_substitutions(substitution_counts, group_size=10):
    # Extract substitution types and counts
    substitutions = list(substitution_counts.keys())
    counts = list(substitution_counts.values())

    # Split substitution types into groups
    grouped_substitutions = [substitutions[i:i+group_size] for i in range(0, len(substitutions), group_size)]
    grouped_counts = [counts[i:i+group_size] for i in range(0, len(counts), group_size)]

    # Pad the shorter groups with zeros
    max_group_len = max(len(group) for group in grouped_counts)
    padded_counts = [group + [0] * (max_group_len - len(group)) for group in grouped_counts]

    # Convert counts to a NumPy array
    counts_array = np.array(padded_counts)

    # Calculate pairwise distances between groups
    group_distances = sch.distance.pdist(counts_array)

    # Perform hierarchical clustering
    # ward method, which minimizes the variance between the clusters.
    #The resulting linkage matrix is used to generate a dendrogram using the sch.dendrogram function.
    linkage_matrix = sch.linkage(group_distances, method='ward')

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram = sch.dendrogram(linkage_matrix, labels=range(1, len(grouped_substitutions)+1), ax=ax)

    # Customize the plot
    ax.set_xlabel('Group')
    ax.set_ylabel('Distance')
    ax.set_title('Substitution Group Clustering')

    # Display the plot
    plt.tight_layout()
    plt.show()

# Call the function with the substitution counts
visualize_substitutions(wer_substitution_counts, group_size=10)

