---
layout : single
author_profile : true
toc: true
author_profile: true
toc_sticky: true
---
# Data Structures Overview  

## Characteristics of Data Structures

| **Data Structure** | **Advantages**                                               | **Disadvantages**                                           |
|--------------------|--------------------------------------------------------------|------------------------------------------------------------|
| **Array**          | Quick insertion, very fast access if index known.            | Slow search, slow deletion, fixed size.                    |
| **Ordered array**  | Quicker search than unsorted array.                          | Slow insertion and deletion, fixed size.                   |
| **Stack**          | Provides last-in, first-out access.                           | Slow access to other items.                                |
| **Queue**          | Provides first-in, first-out access.                          | Slow access to other items.                                |
| **Linked list**    | Quick insertion, quick deletion.                             | Slow search.                                               |
| **Binary tree**    | Quick search, insertion, deletion (if tree remains balanced). | Deletion algorithm is complex.                             |
| **Red-black tree** | Quick search, insertion, deletion. Tree always balanced.      | Complex.                                                   |
| **2-3-4 tree**     | Quick search, insertion, deletion. Tree always balanced. Similar trees good for disk storage. | Complex.                                        |
| **Hash**           | Very fast access table if key known. Fast insertion.         | Slow deletion, access slow if key not known, inefficient memory usage. |
| **Heap**           | Fast insertion, deletion, access to the largest item.         | Slow access to other items.                                |
| **Graph**          | Models real-world situations.                                | Some algorithms are slow and complex.                      |


## Data Structures Comparison

<table>
  <thead>
    <tr>
      <th rowspan="3">Data Structure</th>
      <th colspan="8">Time Complexity</th>
      <th colspan="1">Space Complexity</th>
    </tr>
    <tr>
      <th colspan="4">Average</th>
      <th colspan="4">Worst</th>
      <th colspan="4">Worst</th>
    </tr>
    <tr>
      <th>Access</th>
      <th>Search</th>
      <th>Insertion</th>
      <th>Deletion</th>
      <th>Access</th>
      <th>Search</th>
      <th>Insertion</th>
      <th>Deletion</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Array</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Stack</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Queue</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Singly-Linked List</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Doubly-Linked List</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>  
    </tr>
    <tr>
      <td>Skip List</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(nlog(n))</td>

    </tr>
    <tr>
      <td>Hash Table</td>
      <td>N/A</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>N/A</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Binary Search Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Cartesian Tree</td>
      <td>N/A</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>N/A</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>B-Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Red-Black Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Splay Tree</td>
      <td>N/A</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>N/A</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>AVL Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>KD Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
</tbody>
</table>

## Basic Operations on Data Structures  

The basic operations that are performed on data structures are as follows:
1. **Insertion**: Insertion means addition of a new data element in a data structure.
2. **Deletion**: Deletion means removal of a data element from a data structure if it is found.
3. **Searching**: Searching involves searching for the specified data element in a data structure.
4. **Traversal**: Traversal of a data structure means processing all the data elements present in it.
5. **Sorting**: Arranging data elements of a data structure in a specified order is called sorting.
6. **Merging**: Combining elements of two similar data structures to form a new data
structure of the same type, is called merging.