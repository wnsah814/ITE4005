import sys
import numpy as np

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}
        self.count = 0

def read_dataset(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        attributes = lines[0].strip().split('\t')
        data = [line.strip().split('\t') for line in lines[1:]]
    return attributes, data

def calculate_entropy(data):
    label_counts = {}
    for instance in data:
        label = instance[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    entropy = 0
    total_instances = len(data)
    for label in label_counts:
        probability = label_counts[label] / total_instances
        entropy -= probability * (probability and np.log2(probability))
    return entropy

def calculate_gain_ratio(data, attribute_index):
    attribute_values = {}
    for instance in data:
        attribute_value = instance[attribute_index]
        if attribute_value not in attribute_values:
            attribute_values[attribute_value] = []
        attribute_values[attribute_value].append(instance)
    entropy = calculate_entropy(data)
    remainder = 0
    total_instances = len(data)
    for value in attribute_values:
        probability = len(attribute_values[value]) / total_instances
        remainder += probability * calculate_entropy(attribute_values[value])
    split_info = 0
    for value in attribute_values:
        probability = len(attribute_values[value]) / total_instances
        split_info -= probability * (probability and np.log2(probability))
    if split_info == 0:
        return 0
    gain = entropy - remainder
    gain_ratio = gain / split_info
    return gain_ratio

def choose_best_attribute(data, attributes):
    best_gain_ratio = 0
    best_attribute = None
    for i, attribute in enumerate(attributes[:-1]): # exclude the last attribute: class label
        gain_ratio = calculate_gain_ratio(data, i)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_attribute = attribute
    return best_attribute

def construct_decision_tree(data, attributes):
    labels = [instance[-1] for instance in data]
    # 하나의 class로 구분된 경우
    if len(set(labels)) == 1:
        label = labels[0]
        node = Node(label=label)
        node.count = len(data)
        return node
    
    # 더 이상의 attribute가 없는 경우
    # majority voting
    if len(attributes) == 0:
        label = max(set(labels), key=labels.count)
        node = Node(label=label)
        node.count = len(data)
        return node
    
    best_attribute = choose_best_attribute(data, attributes)
    node = Node(attribute=best_attribute)
    attribute_index = attributes.index(best_attribute)
    attribute_values = set([instance[attribute_index] for instance in data])
    
    for value in attribute_values:
        subset = [instance[:attribute_index] + instance[attribute_index+1:] for instance in data if instance[attribute_index] == value]
        if len(subset) == 0:
            label = max(set(labels), key=labels.count)
            child_node = Node(label=label)
            child_node.count = 0
            node.children[value] = child_node
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.children[value] = construct_decision_tree(subset, new_attributes)
    return node

def classify_instance(instance, tree, attributes):
    if tree.label:
        return tree.label
    
    attribute = instance[attributes.index(tree.attribute)]
    
    # decision tree로 구분할 수 없는 경우 majority voting
    if attribute not in tree.children.keys():    
        majority_label = max(tree.children.values(), key=lambda x: x.count).label
        return majority_label
    return classify_instance(instance, tree.children[attribute], attributes)

def classify_dataset(test_data, tree, attributes):
    results = []
    for instance in test_data:
        result = classify_instance(instance, tree, attributes)
        results.append(result)
    return results

def write_classification_results(output_filename, attributes, test_data, results):
    with open(output_filename, 'w') as file:
        file.write('\t'.join(attributes) + '\n')
        for i, result in enumerate(results):
            file.write(f'\t'.join(test_data[i]) + f'\t{result}\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 2021092379_성준모_hw2.py dt_train.txt dt_test.txt dt_result.txt")
        sys.exit(1)
    
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]

    attributes, training_data = read_dataset(training_file)
    _, test_data = read_dataset(test_file)
    tree = construct_decision_tree(training_data, attributes)
    classification_results = classify_dataset(test_data, tree, attributes)
    write_classification_results(result_file, attributes, test_data, classification_results)