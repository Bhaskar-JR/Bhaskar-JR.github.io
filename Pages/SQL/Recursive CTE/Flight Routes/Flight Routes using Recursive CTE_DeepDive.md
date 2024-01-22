---
layout : single
title : "Flight Routes using Recursive CTE - Deep Dive" 
author_profile: true
toc: true
toc_sticky: true
--- 

# Base Query

## Using correlated Subquery

```sql
  SELECT FlightID, Origin, Destination, 
         CAST(CONCAT(Origin, ',', Destination) AS CHAR(100)) AS Route,
         1 AS Step
  FROM Flights
  WHERE Origin NOT IN (
    SELECT Destination
    FROM Flights AS subFlights
    WHERE subFlights.FlightID = Flights.FlightID
  )
```
  
### Output
  
| FlightID | Origin | Destination | Route | Step |
|----------|--------|-------------|-------|------|
| 1        | A      | B           | A,B   | 1    |
| 2        | D      | E           | D,E   | 1    |
| 3        | F      | G           | F,G   | 1    |
| 4        | M      | D           | M,D   | 1    |
  

## Using Left join

```sql
SELECT
    f1.FlightID,
    f1.Origin,
    f1.Destination,
    CAST(CONCAT(f1.Origin, ',', f1.Destination) AS CHAR(100)) AS Route,
    1 AS Step
FROM
    Flights AS f1
LEFT JOIN
    Flights AS f2 ON f1.Origin = f2.Destination AND f1.FlightID = f2.FlightID
WHERE
    f2.Destination IS NULL
```
  
### Output
  
  
| FlightID | Origin | Destination | Route | Step |
|----------|--------|-------------|-------|------|
| 1        | A      | B           | A,B   | 1    |
| 2        | D      | E           | D,E   | 1    |
| 3        | F      | G           | F,G   | 1    |
| 4        | M      | D           | M,D   | 1    |
  
## First Iteration
  
```sql 
WITH RECURSIVE FlightRoutes AS (
SELECT FlightID, Origin, Destination, CAST(CONCAT(Origin,',',Destination) AS CHAR(100)) As Route, 1 as Step
FROM Flights
WHERE Origin NOT IN (
    SELECT Destination
    FROM Flights AS subFlights
    WHERE subFlights.FlightID = Flights.FlightID
)

UNION ALL
  SELECT fr.FlightID, fr.Origin, t.Destination, CAST(CONCAT(fr.Route, ',', t.Destination) AS CHAR(100)), fr.Step+1 as Step
  FROM FlightRoutes fr
  JOIN Flights t ON fr.FlightID = t.FlightID AND fr.Destination = t.Origin
  WHERE POSITION(t.Destination IN fr.Route) = 0
  AND fr.Step <2

)

select *
from FlightRoutes;
```

### Input to First Iteration

| FlightID | Origin | Destination | Route | Step |
|----------|--------|-------------|-------|------|
| 1        | A      | B           | A,B   | 1    |
| 2        | D      | E           | D,E   | 1    |
| 3        | F      | G           | F,G   | 1    |
| 4        | M      | D           | M,D   | 1    |
  
### Rows getting generated at the First Iteration  
  
| FlightID | Origin | Destination | Route  | Step |
|----------|--------|-------------|--------|------|
| 1        | A      | C           | A,B,C  | 2    |
| 3        | F      | H           | F,G,H  | 2    |
| 4        | M      | A           | M,D,A  | 2    |
  
### Overall Output at the end of first iteration  
  
| FlightID | Origin | Destination | Route   | Step |
|----------|--------|-------------|---------|------|
| 1        | A      | B           | A,B     | 1    |
| 2        | D      | E           | D,E     | 1    |
| 3        | F      | G           | F,G     | 1    |
| 4        | M      | D           | M,D     | 1    |
| 1        | A      | C           | A,B,C   | 2    |
| 3        | F      | H           | F,G,H   | 2    |
| 4        | M      | A           | M,D,A   | 2    |

## Second Iteration

```sql
WITH RECURSIVE FlightRoutes AS (
SELECT FlightID, Origin, Destination, CAST(CONCAT(Origin,',',Destination) AS CHAR(100)) As Route, 1 as Step
FROM Flights
WHERE Origin NOT IN (
    SELECT Destination
    FROM Flights AS subFlights
    WHERE subFlights.FlightID = Flights.FlightID
)

UNION ALL
  SELECT fr.FlightID, fr.Origin, t.Destination, CAST(CONCAT(fr.Route, ',', t.Destination) AS CHAR(100)), fr.Step+1 as Step
  FROM FlightRoutes fr
  JOIN Flights t ON fr.FlightID = t.FlightID AND fr.Destination = t.Origin
  WHERE POSITION(t.Destination IN fr.Route) = 0
  AND fr.Step <3

)

select *
from FlightRoutes;
``` 
  
### Input to Second Iteration
  
| FlightID | Origin | Destination | Route   | Step |
|----------|--------|-------------|---------|------|
| 1        | A      | C           | A,B,C   | 2    |
| 3        | F      | H           | F,G,H   | 2    |
| 4        | M      | A           | M,D,A   | 2    |

### Rows getting generated at the Second Iteration

| FlightID | Origin | Destination | Route     | Step |
|----------|--------|-------------|-----------|------|
| 3        | F      | I           | F,G,H,I   | 3    |
| 4        | M      | N           | M,D,A,N   | 3    |
  
> The newly generated rows will get appended to the continuously expanding table in the recursive CTE.

### Overall Output at the end of second iteration
  
| FlightID | Origin | Destination | Route     | Step |
|----------|--------|-------------|-----------|------|
| 1        | A      | B           | A,B       | 1    |
| 2        | D      | E           | D,E       | 1    |
| 3        | F      | G           | F,G       | 1    |
| 4        | M      | D           | M,D       | 1    |
| 1        | A      | C           | A,B,C     | 2    |
| 3        | F      | H           | F,G,H     | 2    |
| 4        | M      | A           | M,D,A     | 2    |
| 3        | F      | I           | F,G,H,I   | 3    |
| 4        | M      | N           | M,D,A,N   | 3    |

## Extracting the relevant records from the overall table from recursive CTE  
  
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
  
### Output  
  
| FlightID | Origin | Destination | Route     | Step |
|----------|--------|-------------|-----------|------|
| 2        | D      | E           | D,E       | 1    |
| 1        | A      | C           | A,B,C     | 2    |
| 3        | F      | I           | F,G,H,I   | 3    |
| 4        | M      | N           | M,D,A,N   | 3    |



## Summary  
  
> The base query retrieves the initial set of records from the source table based on specified conditions. The recursive portion then takes the output of the base query and generates additional row entries. These new entries are appended to the output of the base query. In each iteration, the input for the next iteration consists of the records generated in the previous iteration. This process continues, and the overall output table grows with each iteration. The number of iterations can be separately tracked, providing a clear view of the flow of input and output across subsequent iterations.