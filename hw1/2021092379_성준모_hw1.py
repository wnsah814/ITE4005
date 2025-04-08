import sys

def read_transactions(input_file):
    transactions = []
    with open(input_file, 'r') as f:
        for line in f:
            items = set(map(int, line.strip().split('\t')))
            transactions.append(items)
    return transactions

def generate_combinations(my_set, k):
    my_set = list(my_set)
    def backtrack(start, cur_comb):
        if len(cur_comb) == k:
            yield cur_comb
            return
        for i in range(start, len(my_set)):
            yield from backtrack(i + 1, cur_comb + [my_set[i]])

    yield from backtrack(0, [])

def is_subset(cand, sets):
    for s in sets:
        if cand.issubset(s):
            return True
    return False

def generate_candidates(frequent_itemsets, pruned_itemsets, k):
    candidates = set()
    for itemset1 in frequent_itemsets:
        for itemset2 in frequent_itemsets:
            for comb in generate_combinations(itemset1.union(itemset2), k):
                if (len(pruned_itemsets) == 0):
                    candidates.add(frozenset(comb))
                else:
                    # prune before candidate generation
                    if not is_subset(set(comb), pruned_itemsets):
                        candidates.add(frozenset(comb))
    return candidates

def calculate_support(itemset, transactions):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return (count / len(transactions)) * 100

def calculate_confidence(rule, transactions):
    antecedent, consequent = rule
    support_antecedent = calculate_support(antecedent, transactions)
    support_antecedent_consequent = calculate_support(antecedent.union(consequent), transactions)
    return (support_antecedent_consequent / support_antecedent) * 100

def generate_association_rules(frequent_itemsets, transactions):
    rules = []
    for itemset, support in frequent_itemsets.items():
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            confidence = calculate_confidence((antecedent, consequent), transactions)
            rules.append((antecedent, consequent, support, confidence))

        for i in range(2, len(itemset)):
            for combination in generate_combinations(itemset, i):
                antecedent = frozenset(combination)
                consequent = itemset - antecedent
                confidence = calculate_confidence((antecedent, consequent), transactions)
                rules.append((antecedent, consequent, support, confidence))

    return rules

def apriori(input_file, output_file, min_support):
    transactions = read_transactions(input_file)
    itemsets = [frozenset([item]) for item in range(min(min(t) for t in transactions), max(max(t) for t in transactions) + 1)]
    k = 2

    frequent_itemsets = {}
    while itemsets:
        accepted_frequent_itemsets = set()
        pruned_frequent_itemsets = set()
        for itemset in itemsets:
            support = calculate_support(itemset, transactions)
            # prune after candidate generation
            if support >= min_support:
                accepted_frequent_itemsets.add(itemset)
                if k > 2: 
                    frequent_itemsets[itemset] = support
            else:
                pruned_frequent_itemsets.add(itemset)

        if not accepted_frequent_itemsets:
            break

        itemsets = generate_candidates(accepted_frequent_itemsets, pruned_frequent_itemsets, k)
        k += 1
    for fis in frequent_itemsets:
        print(set(fis), end=", ")
    # print("freq items", frequent_itemsets)
    rules = generate_association_rules(frequent_itemsets, transactions)
    with open(output_file, 'w') as f:
        for antecedent, consequent, support, confidence in rules:
            f.write(f"{{{','.join(map(str, antecedent))}}}\t{{{','.join(map(str, consequent))}}}\t{support:.2f}\t{confidence:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python apriori.py <min_support> <input_file> <output_file>")
        sys.exit(1)

    min_support = float(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    apriori(input_file, output_file, min_support)
    print("Done")
