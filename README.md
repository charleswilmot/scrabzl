# Usage:


First create dictionaries (takes about 30 secs for a max word length of 7):


```
python create_dictionary.py ../dictionaries/english.txt english --max-word-length 7
```


Start the solving process:

```
python scrabzl.py 5 7 4 ../dictionaries/english_97533_2_7.pkl --no-display --display-freq 1
```

The english dictionary is really huge, and most words in it are very uncommon. This tends to slow down the search.


I recommend to use a dictionary which max word size matches the max size of the grids to be solved:


for a grid of size 5x7, use a dictionary with words of max size 7