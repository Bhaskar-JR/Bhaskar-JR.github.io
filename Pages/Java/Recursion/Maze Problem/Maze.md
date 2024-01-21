---
layout : single
title : Maze Problem
author_profile: true
toc: true
toc_sticky: true
---

**A maze** typically consists of cells. In the context of a maze, a "cell" refers to an individual unit or space within the maze's structure. A maze is generally composed of numerous cells, which can be square, rectangular, or even hexagonal, depending on the design of the maze. These cells are arranged in a specific pattern and are interconnected by pathways, while some cells may be blocked by walls to create the puzzle element of the maze. The goal in a maze is to find a path through these cells from a starting point to an endpoint.

## Rat in a Maze problem
- Movement in four directions  

Print the route/pathways from starting cell to the last cell moving up, down, left, or right, without revisiting any cell in the maze.  
  
![png](Maze.png)  
  
```java
public class FourDirectionMazePath{
    public static void main(String[] args){
        int rows = 3;
        int cols = 3;
        
        if (args.length > 0) {
        try {
            rows = Integer.parseInt(args[0]);
            cols = Integer.parseInt(args[1]);
        } catch (NumberFormatException e) {
            System.out.println("Invalid input. Using default value: 3");
            // rows is already set to 3, so no need to set it again
        }
    }    
    boolean[][] isVisited = new boolean[rows][cols];
    
    print(0,0, rows-1, cols-1,"", isVisited);
        
    }
    
    private static void print(int sr, int sc, int er, int ec,
                              String s, boolean [][] isVisited){
            if(sr<0 || sc<0) return;
            if(sr>er || sc>ec) return;
            if(isVisited[sr][sc] == true ) return;
            if(sr == er && sc ==ec){
                System.out.println(s);
                return;
            }
            
            isVisited[sr][sc] = true;
            
            //go right
            print(sr, sc+1, er, ec, s+ "R", isVisited);
            
            //go down
            print(sr+1, sc, er, ec, s+"D", isVisited);
            
            //go left
            print(sr, sc-1, er,ec,s+"L", isVisited);
            
            //go up
            print(sr-1, sc, er,ec, s+"U", isVisited);
            
            //backtracking
            isVisited[sr][sc] = false;
            }
    }

```  
  
### Code Assessment from Gen AI

The FourDirectionMazePath class in Java that you've provided is designed to print all possible paths through a 3x3 maze, moving up, down, left, or right, without revisiting any cell. Let's review the code in detail:

1.	Class and Main Method:
- The class is correctly defined.
- The main method initializes a 3x3 maze but allows for size customization through command-line arguments.
2.	Command-line Argument Handling:
- The program checks for command-line arguments and tries to parse them as integers for rows and columns.
- If the parsing fails, it catches a NumberFormatException and defaults to a 3x3 maze.
- This is a good practice for handling invalid inputs.
3.	Maze Initialization:
- The maze is represented by a boolean array isVisited, which tracks whether a cell has been visited. This is an appropriate choice for the task.
4.	Recursive print Method:
- The method is designed to explore the maze recursively.
- Base cases are correctly implemented:  
-- It returns if the current cell is out of bounds.  
-- It returns if the current cell is already visited.
-- It prints and returns the path if the end of the maze is reached.
- The method marks the current cell as visited and explores all four directions (right, down, left, up).
- After exploring, it backtracks by marking the current cell as not visited.
5.	Potential Improvements:
- Variable Naming: The variable names sr, sc, er, ec are not immediately clear. More descriptive names like startRow, startCol, endRow, endCol could enhance readability.
