---
layout : single
author_profile : true
toc: true
author_profile: true
toc_sticky: true
---

## Data Structures Comparison

<table>
  <thead>
    <tr>
      <th rowspan="3">Data Structure</th>
      <th colspan="4">Time Complexity</th>
      <th colspan="4">Space Complexity</th>
    </tr>
    <tr>
      <th colspan="4">Average</th>
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
    </tr>
    <tr>
      <td>KD Tree</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))</td>
      <td>O(log(n))
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