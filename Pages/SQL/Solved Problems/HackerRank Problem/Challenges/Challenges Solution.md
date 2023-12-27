---
layout : single
title : HackerRank Challenges Solution
author_profile: true
toc: true
toc_sticky: true
---

## Clarifying the Objective

**Objective**: Select hackers based on the following criteria:
1. **Maximum Challenge Solvers**: Hackers who have solved the highest number of challenges, regardless of ties.
2. **Unique Challenge Solvers**: Hackers who have solved a unique number of challenges compared to others, being the only ones with that specific count.

**Example Scenario**:
- Hackers A and B each solved 7 challenges, the maximum number.
- Hackers D and E are tied with 5 challenges each.
- Hacker F solved 3 challenges, a unique count no other hacker has achieved.

**Expected Outcome**:
- Include Hackers A, B (for solving the maximum challenges) and F (for a unique challenge count).
- Exclude Hackers D and E; their count isn't maximum or unique.

**Summary**: The query aims to identify hackers who are either at the top in challenge count or have a unique count not shared with others.

## Approach

To select hackers who have solved either the maximum or a unique number of challenges, we break down the query into steps:

1. Calculate the number of challenges solved by each hacker.
2. Determine the maximum number of challenges solved by any hacker.
3. Find challenge counts unique to individual hackers.
4. Select hackers who meet either of the above criteria.

**SQL Query Breakdown**:
- **RankedHackers CTE**: Calculates challenges solved by each hacker.
- **MaxChallenges CTE**: Finds the maximum number of challenges solved.
- **UniqueChallenges CTE**: Identifies unique challenge counts.
- **Final SELECT**: Uses LEFT JOIN to filter hackers based on criteria.

## SQL Code using CTE (Common Table Expressions)

```sql
-- Calculate the number of challenges solved by each hacker
WITH ChallengeCounts AS (
    SELECT 
        h.hacker_id, 
        h.name, 
        COUNT(DISTINCT c.challenge_id) AS challengesCount
    FROM 
        Challenges c
    JOIN 
        Hackers h ON c.hacker_id = h.hacker_id
    GROUP BY 
        h.hacker_id, h.name
),
-- Determine the maximum number of challenges solved by any hacker
MaxChallengeCount AS (
    SELECT 
        MAX(challengesCount) AS maxChallenges
    FROM 
        ChallengeCounts
),
-- Find the counts of challenges that are unique to individual hackers
UniqueChallengeCounts AS (
    SELECT 
        challengesCount
    FROM 
        ChallengeCounts
    GROUP BY 
        challengesCount
    HAVING 
        COUNT(*) = 1
)
-- Select hackers who have either solved the maximum number of challenges
-- or have a unique challenge count
SELECT 
    cc.hacker_id, 
    cc.name, 
    cc.challengesCount
FROM 
    ChallengeCounts cc
WHERE 
    cc.challengesCount IN (SELECT maxChallenges FROM MaxChallengeCount)
    OR cc.challengesCount IN (SELECT challengesCount FROM UniqueChallengeCounts)
ORDER BY 
    cc.challengesCount DESC, cc.hacker_id;
```  
  
## SQL Code using Subquery Approach

For compatibility with older versions of MySQL where Common Table Expressions (CTEs) are not supported, an alternative approach using subqueries in the FROM and WHERE clauses can be employed.

### Approach

1. **Joining and Grouping**:
   - Join the `Challenges` and `Hackers` tables.
   - Group the results by hacker to calculate each hacker's challenge count.

2. **Applying Conditions with HAVING**:
   - Use the `HAVING` clause to filter results based on two conditions:
     - The hacker's challenge count matches the maximum number of challenges solved. This is determined by a subquery (`SubMax`) calculating the maximum challenge count across all hackers.
     - The hacker's challenge count is unique, ascertained by another subquery (`SubUnique`). This subquery calculates challenge counts for all hackers and groups them to identify counts that only appear once.

3. **Ordering Results**:
   - Order the results by challenge count in descending order, followed by hacker ID.

### SQL Query

```sql
SELECT 
    h.hacker_id, 
    h.name, 
    COUNT(DISTINCT c.challenge_id) AS challengesCount
FROM 
    Challenges c
JOIN 
    Hackers h ON c.hacker_id = h.hacker_id
GROUP BY 
    h.hacker_id, h.name
HAVING 
    -- Check if the hacker's challenge count is the maximum
    COUNT(DISTINCT c.challenge_id) = (
        SELECT MAX(ChallengeCount) 
        FROM (
            SELECT COUNT(DISTINCT challenge_id) AS ChallengeCount 
            FROM Challenges 
            GROUP BY hacker_id
        ) AS SubMax
    )
    -- OR check if the hacker's challenge count is unique
    OR COUNT(DISTINCT c.challenge_id) IN (
        SELECT ChallengeCount 
        FROM (
            SELECT COUNT(DISTINCT challenge_id) AS ChallengeCount 
            FROM Challenges 
            GROUP BY hacker_id
        ) AS SubUnique
        GROUP BY ChallengeCount 
        HAVING COUNT(*) = 1
    )
ORDER BY 
    COUNT(DISTINCT c.challenge_id) DESC, h.hacker_id;
```

