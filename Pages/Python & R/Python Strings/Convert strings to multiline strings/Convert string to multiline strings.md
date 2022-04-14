---
layout : single
author_profile: true
comments: true
---

# Convert string to multiline strings


```python
# Functions to convert the pairing of unique values and value_counts into text string
# Function to break a word into multiline string of fixed width per line
def paddingString(word, nspaces = 20):
    i = len(word)//nspaces \
        +(len(word)%nspaces > 0)*(len(word)//nspaces > 0)*1 \
        + (len(word)//nspaces == 0)*1
    strA = ""
    for j in range(i-1):
        strA = strA+'\n'*(len(strA)>0)+ word[j*nspaces:(j+1)*nspaces]

    # insert appropriate number of white spaces
    strA = strA + '\n'*(len(strA)>0)*(i>1)+word[(i-1)*nspaces:] \
           + " "*(nspaces-len(word)%nspaces)*(len(word)%nspaces > 0)
    return strA
```


```python
txt1 = "Compelling... written with verve and passion"
print(paddingString(txt1, 15))
```

    Compelling... w
    ritten with ver
    ve and passion
