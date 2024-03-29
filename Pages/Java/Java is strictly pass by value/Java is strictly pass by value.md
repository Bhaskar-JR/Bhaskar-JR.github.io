---
layout : single
author_profile : true
toc: true
author_profile: true
toc_sticky: true
---

## Passing Data and Representing Data Types

### Primitive Data Types and Pass-by-Value

- **Primitive Data Types:** In Java, primitive data types include `int`, `float`, `double`, `char`, `byte`, `short`, `long`, and `boolean`. These types represent single values and not complex objects.

- **Pass-by-Value:** Java is strictly pass-by-value, meaning when you pass a variable to a method, you are passing a copy of its value. In the case of primitives, this value is the actual data.

  For example, if you pass an `int` to a method, a copy of that `int` is made, and any changes to it in the method do not affect the original variable.

### Reference Types (Abstract Data Types) and Their Behavior

- **Reference Types:** In Java, reference types include objects and arrays. They are not primitive types and can refer to complex structures.

- **How They Work:** When you pass a reference type to a method, you pass a copy of the reference, not the actual object. Therefore, you can modify the object the reference points to, but you cannot change the reference to point to a different object.

  For example, if you pass an array or an object to a method, you can change its contents, but if you try to assign a new array or object to that reference, the original outside the method remains unchanged.

### No Pointer Arithmetic

- **No Pointers Like C/C++:** Java does not have pointer types like C/C++. References in Java are similar to pointers in that they refer to objects, but you cannot perform pointer arithmetic. Java abstracts away the complexity of pointers for safety and simplicity.

- **Memory Management:** Java handles memory allocation and garbage collection automatically, meaning programmers don't directly deal with memory addresses or manual memory management.

### Data Structures and Their References

- **Data Structures:** Java provides a rich set of data structures like `ArrayList`, `LinkedList`, `HashMap`, `HashSet`, etc., through its Collections Framework.

- **References to Data Structures:** When working with these data structures, you're handling references to these objects. Any modifications to the data structure via its reference reflect in the original object.

  For example, if you pass an `ArrayList` to a method, modifications to the list (like adding or removing elements) within the method are reflected in the original list.

### Abstract Classes and Interfaces

- **Abstract Classes:** These are classes that cannot be instantiated on their own and are meant to be subclassed. They can include abstract methods (without implementation) that must be implemented by subclasses.

- **Interfaces:** An interface in Java is a completely abstract class; it acts as a contract for what methods a class must implement.

- **Usage:** Both abstract classes and interfaces are used to define a common protocol for a set of classes. They are references to complex types that subclasses or implementers must flesh out.

In summary, Java's design focuses on safety and simplicity, abstracting away complexities like pointer arithmetic and memory management found in languages like C/C++. Understanding the distinction between how primitive and reference types are handled, especially in the context of method calling, is fundamental in Java programming.

### Class-Level Variables in Java

```java
// Class-level variable to keep track of solutions  
private static int count = 0;
```  
  
```java
package nQueens;

import java.util.ArrayList;
import java.util.List;

//Java convention is Pascal case for naming classes
public class NQueensSolver {

	// Class-level variable to keep track of solutions
	private static int count = 0;

	public static void main(String[] args) {

		int n = 5; // board length
		char[][] board = createBoard(n);
		List<char[][]> savedBoards = new ArrayList<>();

		solve(board, 0, savedBoards);
		// print2DArray(board);

	}

	public static void solve(char[][] board, int row, List<char[][]> savedBoards) {

		int n = board.length;
		if (row == n) {
			count += 1;
			System.out.println("State no.:" + count);
			print2DArray(board);
			System.out.println();
                
                // Creating a deep copy of the current board state and adding to savedBoards
                // The addition of deep copy for saving board states
                char[][] clonedBoard = new char[n][n];
                for (int i = 0; i < n; i++) {
                System.arraycopy(board[i], 0, clonedBoard[i], 0, n);
                }
                savedBoards.add(clonedBoard);

			return;
		}

		for (int i = 0; i < n; i++) {
			if (isSafe(row, i, board)) {

				board[row][i] = 'Q';
				// Recursive call
				solve(board, row + 1, savedBoards);

				// backtracking
				board[row][i] = 'X';
			}

		}
	}

	public static boolean isSafe(int row, int col, char[][] board) {

		int n = board.length;

		// same row
		for (int i = 0; i < n; i++) {
			if (board[row][i] == 'Q')
				return false;
		}

		// same column
		for (int j = 0; j < n; j++) {
			if (board[j][col] == 'Q')
				return false;
		}

		// Upper left diagonal
		int p = row;
		for (int q = col; p >= 0 && q >= 0; p--, q--) {
			if (board[p][q] == 'Q')
				return false;
		}

		// Upper right diagonal
		p = row;
		for (int q = col; p >= 0 && q < n; p--, q++) {
			if (board[p][q] == 'Q')
				return false;
		}

		// Lower right diagonal
		p = row;
		for (int q = col; p < n && q < n; p++, q++) {
			if (board[p][q] == 'Q')
				return false;
		}

		// Lower left diagonal
		p = row;
		for (int q = col; p < n && q >= 0; p++, q--) {
			if (board[p][q] == 'Q')
				return false;
		}

		return true;

	}

	// Method to create board
	public static char[][] createBoard(int size) {
		char[][] board = new char[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				board[i][j] = 'X'; // Initialize with a default character, e.g., a dash
			}
		}
		return board;
	}

	// Method to print a 2D array
	public static void print2DArray(char[][] array) {
		for (char[] row : array) {
			for (char element : row) {
				System.out.print(element + " ");
			}
			System.out.println(); // Print a new line at the end of each row
		}
	}
}
```



Creating a class-level variable, such as the `count` variable in above example, serves the purpose of maintaining state or shared information across multiple method calls within the same class. Here are some reasons why you might want to use a class-level variable:

1. **Shared State Across Methods:**
   The class-level variable allows different methods within the class to access and modify a shared piece of data. In above example, the `count` variable is incremented and used across the main method and the `solve` method.

2. **Preserving State Between Method Calls:**
   Class-level variables retain their values between method calls. In scenarios where you need to preserve information or state across multiple invocations of methods, a class-level variable is useful. In your code, the `count` variable accumulates the number of solutions across different recursive calls to the `solve` method.

3. **Avoiding Redundant Parameters:**
   Instead of passing the same piece of information as a parameter to multiple methods, a class-level variable eliminates the need for redundant parameters. This can make the code cleaner and more concise.

4. **Centralized Management of Shared Data:**
   When multiple methods need access to a shared piece of data, having it as a class-level variable centralizes its management. Changes to the shared data are visible and consistent across all methods.

5. **Global Scope Within the Class:**
   Class-level variables have a scope that spans the entire class, providing a form of "global" access within that class. This can be advantageous when you want to maintain a single point of control for a specific piece of information.
