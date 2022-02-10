d_list = ['convention', 'python', 'groups', 'class', 'ass']


def del_letter(word, verbose=False):
    delete_l = []
    split_l = []

    for indexer in range(len(word)):
        split_l.append((word[:indexer], word[indexer:]))

    for start, ender in split_l:
        delete_l.append(start + ender[1:])

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


def switch_letter(word, verbose=False):
    switch_l = []
    split_l = []

    for indexer in range(len(word)):
        split_l.append((word[:indexer], word[indexer:]))
    switch_l = [tyt + bibbob[1] + bibbob[0] + bibbob[2:] for tyt, bibbob in split_l if len(bibbob) >= 2]

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


w = 'python3'
del_l, switch_l = del_letter(w, verbose=True), switch_letter(w, verbose=True)
for del_word, switch_word in zip(del_l, switch_l):
    if del_word in d_list:
        print(f"Delete Match: {del_word}")
    if switch_word in d_list:
        print(f"Switch Match: {switch_word}")

