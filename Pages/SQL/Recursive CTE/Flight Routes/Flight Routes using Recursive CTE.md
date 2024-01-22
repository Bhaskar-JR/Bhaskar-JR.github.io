---
layout : single
title : "Flight Routes using Recursive CTE" 
author_profile: true
toc: true
toc_sticky: true
---  

# Flight Routes Problem  
  
## Database Source Table 

| FlightID | Origin | Destination |
|----------|--------|-------------|
| 1        | A      | B           |
| 1        | B      | C           |
| 2        | D      | E           |
| 3        | F      | G           |
| 3        | G      | H           |
| 3        | H      | I           |
| 4        | A      | N           |
| 4        | M      | D           |
| 4        | D      | A           |


## Desired Output table  

| FlightID | Origin | Destination | Route     | Step |
|----------|--------|-------------|-----------|------|
| 2        | D      | E           | D,E       | 1    |
| 1        | A      | C           | A,B,C     | 2    |
| 3        | F      | I           | F,G,H,I   | 3    |
| 4        | M      | N           | M,D,A,N   | 3    |
  
# How the Solution Works

> The base query will fetch the relevant records from the source table using the conditions specified.
> It will then call then call the recursive portion of the query.
> The first iteration will take the output of the base query as the input and generate row entries.
> These row entries will be appended to the output of the base query.
> The second iteration will take the generated row entries/records of the first iteration as input.
>The inputs to the second iteration will only the records fetched as output from the first iteration.  
> Meanwhile, that the overall output table would will continue to update with rows from each iteration getting appended.
> We can declare/separately track the number of iterations.
> It is easy to appreciate the flow of input, output across subsequent iterations and the continuously growing target table through the concrete example.  
  


# Flights Table
We will start from scratch in MySQL.  
Let's create a Flights table and populate it with dummy data.

```sql
SHOW tables;
SELECT * from Flights;
DROP TABLE Flights;
```
  
```sql
-- Create the table
CREATE TABLE Flights (
    FlightID INT,
    Origin CHAR(1),
    Destination CHAR(1)
);
```

```sql
-- Insert data into the table
INSERT INTO Flights (FlightID, Origin, Destination) VALUES
(1, 'A', 'B'),
(1, 'B', 'C'),
(2, 'D', 'E'),
(3, 'F', 'G'),
(3, 'G', 'H'),
(3, 'H', 'I'),
(4, 'A', 'N'),
(4, 'M', 'D'),
(4, 'D', 'A');
```
  
**Results of the Insert Query**  
  
| FlightID | Origin | Destination |
|----------|--------|-------------|
| 1        | A      | B           |
| 1        | B      | C           |
| 2        | D      | E           |
| 3        | F      | G           |
| 3        | G      | H           |
| 3        | H      | I           |
| 4        | A      | N           |
| 4        | M      | D           |
| 4        | D      | A           |

# Solution using correlated Subquery 

```sql
WITH RECURSIVE FlightRoutes AS (

  -- Anchor Member
  SELECT FlightID, Origin, Destination, 
         CAST(CONCAT(Origin, ',', Destination) AS CHAR(100)) AS Route,
         1 AS Step
  FROM Flights
  WHERE Origin NOT IN (
    SELECT Destination
    FROM Flights AS subFlights
    WHERE subFlights.FlightID = Flights.FlightID
  )

  UNION ALL
  
  -- Recursive Member
  SELECT fr.FlightID, fr.Origin, t.Destination, 
              CAST(CONCAT(fr.Route, ',', t.Destination) AS CHAR(100)), 
              fr.Step + 1 AS Step
  FROM FlightRoutes fr
  JOIN Flights t ON fr.FlightID = t.FlightID AND fr.Destination = t.Origin
  WHERE POSITION(t.Destination IN fr.Route) = 0
    AND fr.Step < 3
)

SELECT
  FlightID,
  Origin,
  Destination AS FinalDestination,
  Route,
  Step AS CountOfConnectingFlights
FROM (
  SELECT
    FlightID,
    Origin,
    Destination,
    Route,
    Step,
    ROW_NUMBER() OVER (PARTITION BY FlightID ORDER BY FlightID, Step DESC) AS RowNumber
  FROM FlightRoutes
) AS tbl
WHERE RowNumber = 1;
```  

## Output  

| FlightID | Origin | Destination | Route     | Step |
|----------|--------|-------------|-----------|------|
| 2        | D      | E           | D,E       | 1    |
| 1        | A      | C           | A,B,C     | 2    |
| 3        | F      | I           | F,G,H,I   | 3    |
| 4        | M      | N           | M,D,A,N   | 3    |  


# Solution using correlated subquery and with additional check columns


```sql
WITH RECURSIVE FlightRoutes AS (

-- Anchor Member
SELECT FlightID, Origin, Destination, CAST(CONCAT(Origin,',',Destination) AS CHAR(100)) As Route, 1 as Step
FROM Flights
WHERE Origin NOT IN (
    SELECT Destination
    FROM Flights AS subFlights
    WHERE subFlights.FlightID = Flights.FlightID
)

UNION ALL

-- Recursive Member
  SELECT fr.FlightID, fr.Origin, t.Destination, CAST(CONCAT(fr.Route, ',', t.Destination) AS CHAR(100)), fr.Step+1 as Step
  FROM FlightRoutes fr
  JOIN Flights t ON fr.FlightID = t.FlightID AND fr.Destination = t.Origin
  
  -- Terminating condition to avoid infinite recursion
  WHERE POSITION(t.Destination IN fr.Route) = 0
  AND fr.Step <2

)

select FlightID, 
       Origin,
       Destination as FinalDestination,
       Route, 
       Step As CountOfConnectingFlights, 
       SUBSTRING_INDEX(Route, ',', 1) as Origin_check,
       SUBSTRING_INDEX(Route, ',', -1) as FinalDestination_check
From (Select FlightID, Origin, Destination, Route, Step,
ROW_NUMBER() OVER ( PARTITION BY FlightID
ORDER BY FlightID, Step DESC) as RowNumber
from FlightRoutes) as tbl
where RowNumber = 1;
```

## Output  
  
| FlightID | Origin | Destination | Route     | Step | Origin_check | FinalDestination_Check |
|----------|--------|-------------|-----------|------|--------------|------------------------|
| 2        | D      | E           | D,E       | 1    | D            | E                      |
| 1        | A      | C           | A,B,C     | 2    | A            | C                      |
| 3        | F      | I           | F,G,H,I   | 3    | F            | I                      |
| 4        | M      | N           | M,D,A,N   | 3    | M            | N                      |
  
## Solution Refactored with joins instead of correlated subquery in Base query   
  
Correlated subqueries are computationally inefficient as they require re-running the subquery for each record in the dataset.  
  
Hence, we will be refactoring our code and utilise **joins** to achieve same outcome.  
  
```sql
WITH RECURSIVE FlightRoutes AS (
-- base query/ anchor member
SELECT
    f1.FlightID,
    f1.Origin,
    f1.Destination,
    CAST(CONCAT(f1.Origin, ',', f1.Destination) AS CHAR(100)) AS Route,
    1 AS Step
FROM
    Flights AS f1
LEFT JOIN
    Flights AS f2 
    ON 
    f1.Origin = f2.Destination 
    AND
    f1.FlightID = f2.FlightID
WHERE
    f2.Destination IS NULL

UNION ALL

-- recursive part
  SELECT fr.FlightID,
		 fr.Origin, 
		 t.Destination, 
		CAST(CONCAT(fr.Route, ',', t.Destination) AS CHAR(100)), 
        fr.Step+1 as Step
  FROM FlightRoutes fr
  JOIN Flights t 
  ON 
  fr.FlightID = t.FlightID 
  AND 
  fr.Destination = t.Origin
  
  -- terminating condition
  WHERE POSITION(t.Destination IN fr.Route) = 0

)

select FlightID, 
       Origin,
       Destination as FinalDestination,
       Route, 
       Step As CountOfConnectingFlights, 
       SUBSTRING_INDEX(Route, ',', 1) as Origin_check,
       SUBSTRING_INDEX(Route, ',', -1) as FinalDestination_check
From (Select FlightID, Origin, Destination, Route, Step,
ROW_NUMBER() OVER ( PARTITION BY FlightID
ORDER BY FlightID, Step DESC) as RowNumber
from FlightRoutes) as tbl
where RowNumber = 1;

```  
  
## Output  
  
| FlightID | Origin | Destination | Route     | Step | Origin_check | FinalDestination_Check |
|----------|--------|-------------|-----------|------|--------------|------------------------|
| 2        | D      | E           | D,E       | 1    | D            | E                      |
| 1        | A      | C           | A,B,C     | 2    | A            | C                      |
| 3        | F      | I           | F,G,H,I   | 3    | F            | I                      |
| 4        | M      | N           | M,D,A,N   | 3    | M            | N                      |  
  


