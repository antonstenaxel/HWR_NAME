import sys

def min_edit_distance(s1, s2):
    if len(s1) < len(s2):
        return min_edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1

            if c1==c2:
                substitutions = previous_row[j]
            else:
                substitutions = previous_row[j] + 2     # Levenshtein Substitution cost for

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def main():
    source = sys.argv[1]
    target = sys.argv[2]

    # print(target, source)

    print("Minimum edit distance = ", min_edit_distance(source, target))

if __name__ == '__main__':
    main()

