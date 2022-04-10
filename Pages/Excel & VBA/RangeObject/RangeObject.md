---
layout : single
author_profile : true
toc_sticky : true
toc : true
---

# Introduction to Range objects   

## Why Ranges ? Whats the big deal about it !!
Manipulating Ranges are fundamental to data wrangling in Excel. After all the data is trapped inside cells with each cell having a distinct row number and column number. Understanding of ranges will help us in creating smart formulas and perform   
a sequence of tasks with ease. Even if one does not use VBA, it is a topic that would infact enhance our data handling capabilities.And for VBA, it goes without saying that range objects are the basic building blocks.  

## Diving into Ranges  

A Range is either a single cell or collection of multiple cells. Now this collection of cells can either be in a single contiguous block or can be non-contiguous. **Contiguous** means the cells are within a rectangular grid or within a rectangular block say having x number of rows and y number of columns. So, a **contiguous range is a range with a fixed rectangular boundary**, Whereas non-contiguous range can have any shape. If we see visually, we can have blocks of cells spread across the worksheet. It would not conform to single rectangular block. Since, in most of the cases **we will be dealing with contiguous ranges**. We will be limiting our scope of discussion to contiguous ranges.  

We are broadly going to cover **Range declaration and Assignment**, **Range.Cells property, Range.Offset method, Range.resize method.** We will understand in more detail, try few variations by changing arguments, parameters that I shall pass into these properties. To change the arguments and also extend the utility of these properties, I shall take help of the **readonly range properties** such as Range.rows.columns, Range.columns.count, Range.Address, Range.Row, Range.Column.


<iframe width="504" height="284" src="https://www.youtube.com/embed/9xZAFKvwh0o" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  



# Range Cells Property  

Range Cells   


<iframe width="560" height="315" src="https://www.youtube.com/embed/cFdIPkdWVhk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  


# Range Offset Method  

Offsetting   

<iframe width="560" height="315" src="https://www.youtube.com/embed/kANKKoN0zj4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   



# Range Resize Method  

<iframe width="560" height="315" src="https://www.youtube.com/embed/QrpS4C-Gmds" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Range Readonly properties  

<iframe width="950" height="534" src="https://www.youtube.com/embed/bXx5rEnX0HU?list=PLeE_zyX_pEtnqxy3pZp6kiUwC8OKpGgS0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
