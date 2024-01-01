---
layout : single
title : Filtering data
toc: true
author_profile: true
toc_sticky: true
---
  
## SQL Clause Usage and Best Practices

### 1. Using the WHERE Clause:
- **Primary Use:** Filters rows before any grouping or aggregation occurs. It's applied directly to the raw data in the tables.
- **Common Operators:** Includes =, !=, >, <, >=, <=, BETWEEN, IN, IS NULL, IS NOT NULL, LIKE, and others.
- **Filtering with Subqueries:** Allows complex conditions using subqueries, which can return a list of values, a single value, or even perform existential checks with EXISTS.
- **Note:** Cannot be used to filter aggregated data (like sums or averages).

### 2. Using the FROM Clause (specifically in Joins with the ON Clause):
- **Role in Joins:** The ON clause is part of the syntax for joins (e.g., INNER JOIN, LEFT JOIN) and specifies the conditions for how rows from different tables should be matched.
- **Filtering Aspect:** While primarily used for specifying join conditions, it inherently filters data by determining which rows from the joined tables meet the join condition.
- **Note:** The ON clause is not a general-purpose filtering tool like WHERE but is specific to defining relationships between tables in a join.

### 3. Using the HAVING Clause:
- **Primary Use:** Filters groups of rows after the GROUP BY operation.
- **Applicability:** Used when you have aggregations (SUM, COUNT, AVG, etc.) in your SELECT statement and want to apply conditions on these aggregated results.
- **Difference from WHERE:** WHERE filters individual rows before grouping, while HAVING filters groups after grouping.
- **Note:** Often used in conjunction with GROUP BY, but can be used without it if the query involves aggregate functions.

### Additional Points:
- **Order of Execution:** WHERE -> GROUP BY -> HAVING -> SELECT. This is the logical processing order, not necessarily the physical execution order used by the SQL engine.
- **Performance Considerations:** Using WHERE to filter as much as possible before aggregation can improve query performance. HAVING should be used for conditions that cannot be applied before aggregation.

### WHERE Versus HAVING – CAUTION AND TAKEAWAYS
The purpose of both clauses is to filter data. If you are trying to: 
- Filter on particular columns, write your conditions within the WHERE clause.
- Filter on aggregations, write your conditions within the HAVING clause.

The contents of a WHERE and HAVING clause cannot be swapped: 
- Never put a condition with an aggregation in the WHERE clause. You will get an error.
- Never put a condition in the HAVING clause that does not involve an aggregation. Those conditions are evaluated much more efficiently in the WHERE clause.

### Filtering data – Concise Summary with Sample codes
- **using WHERE clause:**
  - filtering on columns, using =, BETWEEN, IN, IS NULL, LIKE.
  - filtering on subqueries.
- **using FROM clause:** When joining together tables, the ON clause specifies how they should be linked together. This is where you can include conditions to restrict rows of data returned by the query.
- **using HAVING clause:** If there are aggregations within the SELECT statement, the HAVING clause is where you specify how the aggregations should be filtered.

#### References:
- SQL Pocket Guide by Alice Zhou.
- “SQL for Data Scientists - A Beginner's Guide for Building Datasets for Analysis” by Renee M. P. Teate (Chapter 3 - The WHERE Clause for the sample codes in this section.)
  
#### Sample Code
  
-- Filtering Using predicate on column values within Where clause
```sql
SELECT
    market_date,
    customer_id,
    vendor_id,
    quantity * cost_to_customer_per_qty AS price
FROM farmers_market.customer_purchases
WHERE
    customer_id = 4
    AND vendor_id = 7;
```

-- Filtering Using BETWEEN in WHERE clause  
```sql
SELECT *
FROM farmers_market.vendor_booth_assignments
WHERE
    vendor_id = 7 
    AND market_date BETWEEN '2019-03-02' and '2019-03-16'
ORDER BY market_date;
```

-- Filtering Using IN condition in WHERE clause
```sql  
SELECT
    customer_id,
    customer_first_name,
    customer_last_name
FROM farmers_market.customer
WHERE
    customer_first_name IN ('Renee', 'Rene', 'Renée', 'René', 'Renne');
```
  
-- Filtering Using Subquery within Where clause  
```sql
-- Query: Customer Purchases on Rainy Market Dates
SELECT
    market_date,
    customer_id,
    vendor_id,
    quantity * cost_to_customer_per_qty AS price
FROM
    farmers_market.customer_purchases
WHERE
    market_date IN (
        SELECT market_date
        FROM farmers_market.market_date_info
        WHERE market_rain_flag = 1
    )
LIMIT 5;
```
-- Filtering using HAVING  
```sql
-- Query: Vendor Inventory Analysis with Filtering using Having  
SELECT
    vendor_id,
    COUNT(DISTINCT product_id) AS different_products_offered,
    SUM(quantity * original_price) AS value_of_inventory,
    SUM(quantity) AS inventory_item_count,
    SUM(quantity * original_price) / SUM(quantity) AS average_item_price
FROM
    farmers_market.vendor_inventory
WHERE
    market_date BETWEEN '2019-03-02' AND '2019-03-16'
GROUP BY
    vendor_id
HAVING
    inventory_item_count >= 100
ORDER BY
    vendor_id;
```
**Note**, the SQL engine is designed to recognize and appropriately handle aliases from the SELECT clause in the HAVING and ORDER BY clauses, despite the logical processing order of SQL queries. 