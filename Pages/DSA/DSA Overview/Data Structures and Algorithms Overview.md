---
layout : single
author_profile : true
toc: true
author_profile: true
toc_sticky: true
---

## Data Structures Comparison

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    table {
      border-collapse: collapse;
      width: 100%;
    }

    th, td {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }

    th {
      background-color: #f2f2f2;
    }
  </style>
</head>
<body>

<table>
  <thead>
    <tr>
      <th rowspan="2">Data Structure</th>
      <th colspan="3">Average Case</th>
      <th colspan="3">Worse Case</th>
    </tr>
    <tr>
      <th>Search</th>
      <th>Insert</th>
      <th>Delete</th>
      <th>Search</th>
      <th>Insert</th>
      <th>Delete</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Array</td>
      <td colspan="2">O(n)</td>
      <td colspan="3">O(n)</td>
    </tr>
    <tr>
      <td>Sorted Array</td>
      <td>O(log n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(log n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Linked List</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
    </tr>
    <tr>
      <td>Doubly Linked List</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
    </tr>
    <tr>
      <td>Stack</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(1)</td>
      <td>O(1)</td>
    </tr>
    <tr>
      <td>Hash table</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(1)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>Binary Search Tree</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
      <td>O(n)</td>
    </tr>
    <tr>
      <td>B-Tree</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
    </tr>
    <tr>
      <td>Red-Black tree</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
    </tr>
    <tr>
      <td>AVL Tree</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
      <td>O(log n)</td>
    </tr>
  </tbody>
</table>

</body>
</html>
