---
layout : single
title : HackerRank Contest Leaderboard Solution
author_profile: true
toc: true
toc_sticky: true
---
Going back to the [problem](/Contest Leaderboard.md)  
  
### SQL Code using CTE Approach

This SQL script is designed to calculate the maximum score for each challenge per hacker using a Common Table Expression (CTE) and then summing these scores to determine their total score.

#### CTE: Calculating Maximum Score Per Challenge
```sql
-- CTE to calculate the maximum score for each challenge per hacker
WITH t AS (
    SELECT 
        h.hacker_id,             -- Hacker's ID
        h.name,                  -- Hacker's Name
        c.challenge_id,          -- ID of the challenge
        MAX(c.score) AS score    -- Maximum score per challenge for each hacker
    FROM 
        Hackers h                -- Hackers table
        JOIN Submissions c ON h.hacker_id = c.hacker_id 
        -- Joining with Submissions table
    GROUP BY 
        h.hacker_id, h.name, c.challenge_id 
        -- Grouping by hacker_id, name, and challenge_id
)
-- Main query to sum up the maximum scores of each hacker
SELECT 
    t.hacker_id,                -- Hacker's ID
    t.name,                     -- Hacker's Name
    SUM(t.score) AS totalscore  -- Total score for each hacker
FROM 
    t                           -- Using the CTE defined above
GROUP BY
    t.hacker_id,                -- Grouping by hacker_id
    t.name                      
    -- and name to sum scores for each unique hacker
HAVING 
    SUM(t.score) > 0            
    -- Filtering out hackers with zero or negative total scores
ORDER BY 
    totalscore DESC,            -- Ordering by total score in descending order
    t.hacker_id ASC;            -- and then by hacker_id in ascending order for ties
```  
  
This query performs the following actions:

1. **CTE 't' Computation**: The Common Table Expression (CTE) named `t` computes the maximum score obtained by each hacker for each challenge. This CTE groups by `hacker_id`, `name`, and `challenge_id`, ensuring the calculation of the maximum score for each individual challenge that each hacker participated in.

2. **Summing Maximum Scores**: The main query sums these maximum scores for each hacker across all challenges. This aggregated sum represents the total score for each hacker.

3. **Applying HAVING Clause**: The query includes a `HAVING` clause to filter out any hackers whose total score is 0 or less. The focus is on those who have a positive total score.

4. **Ordering Results**: Finally, the results are ordered by `totalscore` in descending order. This means the hacker with the highest total score is listed first. In the event of ties in the total score, `hacker_id` is used to sort these results in ascending order.
  
### SQL Code using Subquery Approach

This SQL query uses a subquery to determine the maximum score per challenge for each hacker and then sums these scores to calculate their total score.

#### Main Query: Summing Scores and Ordering
```sql
SELECT 
    sub.hacker_id,             -- Selecting hacker's ID
    sub.name,                  -- Selecting hacker's name
    SUM(sub.score) AS totalscore 
    -- Summing up the scores to get total score
FROM (
    -- Subquery to get maximum score per challenge for each hacker
    SELECT 
        h.hacker_id,             -- Hacker's ID
        h.name,                  -- Hacker's name
        c.challenge_id,          -- Challenge ID
        MAX(c.score) AS score    
        -- Maximum score for each challenge per hacker
    FROM 
        Hackers h                -- From Hackers table
        JOIN Submissions c ON h.hacker_id = c.hacker_id 
        -- Joining with Submissions table
    GROUP BY 
        h.hacker_id, h.name, c.challenge_id 
        -- Grouping by hacker_id, name, and challenge_id
) AS sub                          -- Alias for the subquery
GROUP BY
    sub.hacker_id,                 -- Grouping results by hacker_id
    sub.name                       -- and name in the outer query
HAVING 
    SUM(sub.score) > 0             
    -- Filtering to include only hackers with a positive total score
ORDER BY 
    totalscore DESC,               -- Ordering by total score in descending order
    sub.hacker_id ASC;             -- and then by hacker_id in ascending order for ties
```