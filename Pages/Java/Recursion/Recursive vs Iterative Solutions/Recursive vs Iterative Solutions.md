---
layout : single
author_profile : true
toc: true
author_profile: true
toc_sticky: true
toc_label: "Recursive Vs. Iterative Solutions"
---

Source : This section has been reproduced from the book "**DSA - Annotated Reference with Examples**" by Granville Barnett, Luca Del Tongo.  
  
One of the most succinct properties of modern programming languages like C++, C#, and Java (as well as many others) is that these languages allow you to deﬁne methods that reference themselves, such methods are said to be recursive. One of the biggest advantages recursive methods bring to the table is that they usually result in more readable, and compact solutions to problems.  
  
A recursive method then is one that is deﬁned in terms of itself. Generally a recursive algorithms has two main properties:  

1. One or more base cases; and
2. A recursive case

For now we will brieﬂy cover these two aspects of recursive algorithms. With each recursive call we should be making progress to our base case otherwise we are going to run into trouble. The trouble we speak of manifests itself typically as a stack overﬂow, we will describe why later.  
  
Now that we have brieﬂy described what a recursive algorithm is and why you might want to use such an approach for your algorithms we will now talk about iterative solutions. An iterative solution uses no recursion whatsoever. An iterative solution relies only on the use of loops (e.g. for, while, do-while, etc). The down side to iterative algorithms is that they tend not to be as clear as to their recursive counterparts with respect to their operation. The major advantage of iterative solutions is speed. Most production software you will ﬁnd uses little or no recursive algorithms whatsoever. The latter property can sometimes be a companies prerequisite to checking in code, e.g. upon checking in a static analysis tool may verify that the code the developer is checking in contains no recursive algorithms. Normally it is systems level code that has this zero tolerance policy for recursive algorithms.  
  
Using recursion should always be reserved for fast algorithms, you should avoid it for the following algorithm run time deﬁciencies:

1. O(n^2)
2. O(n^3)
3. O(2^n)


If you use recursion for algorithms with any of the above run time eﬃciency’s you are inviting trouble. The growth rate of these algorithms is high and in most cases such algorithms will lean very heavily on techniques like divide and conquer. While constantly splitting problems into smaller problems is good practice, in these cases you are going to be spawning a lot of method calls. All this overhead (method calls don’t come that cheap) will soon pile up and either cause your algorithm to run a lot slower than expected, or worse, you will run out of stack space. When you exceed the allotted stack space for a thread the process will be shutdown by the operating system. This is the case irrespective of the platform you use, e.g. .NET, or native C++ etc. You can ask for a bigger stack size, but you typically only want to do this if you have a very good reason to do so.  

## C.2 Some problems are recursive in nature

Something that you may come across is that some data structures and algorithms are actually recursive in nature. A perfect example of this is a tree data structure. A common tree node usually contains a value, along with two pointers to two other nodes of the same node type. As you can see tree is recursive in its makeup wit each node possibly pointing to two other nodes.  
  
When using recursive algorithms on tree’s it makes sense as you are simply adhering to the inherent design of the data structure you are operating on. Of course it is not all good news, after all we are still bound by the limitations we have mentioned previously in this chapter.  
  
We can also look at sorting algorithms like merge sort, and quick sort. Both of these algorithms are recursive in their design and so it makes sense to model them recursively.

## C.3 Summary

Recursion is a powerful tool, and one that all programmers should know of. Often software projects will take a trade between readability, and eﬃciency in which case recursion is great provided you don’t go and use it to implement an algorithm with a quadratic run time or higher. Of course this is not a rule of thumb, this is just us throwing caution to the wind. Defensive coding will always prevail.  
  
Many times recursion has a natural home in recursive data structures and algorithms which are recursive in nature. Using recursion in such scenarios is perfectly acceptable. Using recursion for something like linked list traversal is a little overkill. Its iterative counterpart is probably less lines of code than its recursive counterpart.  
  
Because we can only talk about the implications of using recursion from an abstract point of view you should consult your compiler and run time environment for more details. It may be the case that your compiler recognises things like tail recursion and can optimise them. This isn’t unheard of, in fact most commercial compilers will do this. The amount of optimisation compilers can do though is somewhat limited by the fact that you are still using recursion. You, as the developer have to accept certain accountability’s for performance.