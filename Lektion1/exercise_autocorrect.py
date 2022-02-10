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


# doesnt work
def auto_correct(word):
    pl = []
    del_l, switch_l = del_letter(word, verbose=True), switch_letter(word, verbose=True)
    for del_word, switch_word in zip(del_l, switch_l):
        if del_word in d_list:
            print('1', del_word)
            pl.append(del_word)
        if switch_word in d_list:
            pl.append(switch_word)
            print('2', switch_word)
    return pl


def auto_correct2(word):
    pl = []
    del_l, switch_l = del_letter(word, verbose=True), switch_letter(word, verbose=True)
    for del_word in del_l:
        for d in d_list:
            if del_word == d:
                pl.append(del_word)
    # same scenario for switch
    # takes too much time looping through lists if it's too long
    return pl


print(auto_correct(w))
