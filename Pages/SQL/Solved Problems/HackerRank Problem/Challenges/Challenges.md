---
layout : single
classes: wide
title : HackerRank Problem
author_profile: true
---

### HackerRank Problem: Challenges | Medium Level
[Problem Link](https://www.hackerrank.com/challenges/challenges/problem)

**Description**:  
Julia asked her students to create some coding challenges. The task is to write a query that prints the `hacker_id`, `name`, and the total number of challenges created by each student. The results should be sorted by the total number of challenges in descending order. If more than one student created the same number of challenges, sort the results by `hacker_id`. Exclude students from the results if they created the same number of challenges as others but less than the maximum number of challenges created.

**Input Format**:  
The input comes from two tables containing challenge data:

1. **Hackers**:
   - `hacker_id` (Integer): The id of the hacker.
   - `name` (String): The name of the hacker.

   | Column     | Type    |
   |------------|---------|
   | hacker_id  | Integer |
   | name       | String  |

2. **Challenges**:
   - `challenge_id` (Integer): The id of the challenge.
   - `hacker_id` (Integer): The id of the student who created the challenge.

   | Column        | Type    |
   |---------------|---------|
   | challenge_id  | Integer |
   | hacker_id     | Integer |
  
For solution, [continue](/Challenges Solution.md)