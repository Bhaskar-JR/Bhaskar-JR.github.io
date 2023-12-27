---
layout : single
classes: wide
title : HackerRank Problem
author_profile: true
---

## HackerRank Problem: Contest Leaderboard  
Difficulty : Medium Level  
[Problem Link](https://www.hackerrank.com/challenges/contest-leaderboard/problem)

**Description**:  
Julia has tasked you with another coding contest challenge. The goal is to calculate the total score of each hacker, which is the sum of their maximum scores for all challenges. You need to write a query that prints the `hacker_id`, `name`, and total score of the hackers, ordering them by their descending score. In cases where multiple hackers have the same total score, sort them by ascending `hacker_id`. Exclude any hackers with a total score of 0 from your result.

**Input Format**:  
The input consists of data from two tables:

1. **Hackers**:
   - `hacker_id` (Integer): The id of the hacker.
   - `name` (String): The name of the hacker.

   | Column     | Type    |
   |------------|---------|
   | hacker_id  | Integer |
   | name       | String  |

2. **Submissions**:
   - `submission_id` (Integer): The id of the submission.
   - `hacker_id` (Integer): The id of the hacker who made the submission.
   - `challenge_id` (Integer): The id of the challenge for which the submission belongs.
   - `score` (Integer): The score of the submission.

   | Column         | Type    |
   |----------------|---------|
   | submission_id  | Integer |
   | hacker_id      | Integer |
   | challenge_id   | Integer |
   | score          | Integer |
  
For solutions, [continue](/Contest Leaderboard Solutions.md).