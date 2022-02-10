"""
You use autocorrect every day on your cell phone and computer. In this assignment, you will explore what really goes on behind the scenes.

Below you find code snippets that might be helpful as you build your make your own autocorrect.
"""

def convert(lst):
    return ' '.join(lst).split()
     
 
# Start with a list of words.
lst =  ['Hello Geeks for geeks']
print( convert(lst))

mylist = ['hello', 'hola', 'hi, how are you today ?']

myword = 'today'

print(any(myword in w for w in mylist))

# Delete a letter
def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words 
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''
    
    delete_l = []
    split_l = []
    
    for indexer in range(len(word)):
        split_l.append((word[:indexer],word[indexer:]))
        
    for start,ender in split_l:
        delete_l.append(start+ender[1:])   
    
    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return  delete_l


delete_word_l = delete_letter(word="holiday",
                        verbose=True)

# Switch letter
def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    ''' 
    
    switch_l = []
    split_l = []
    
    for indexer in range(len(word)):
        split_l.append((word[:indexer],word[indexer:]))
    switch_l = [tyt + bibbob[1] + bibbob[0] + bibbob[2:] for tyt,bibbob in split_l if len(bibbob) >= 2]    
    
    
    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}") 
    
    return switch_l


switch_word_l = switch_letter(word="eta",
                         verbose=True)