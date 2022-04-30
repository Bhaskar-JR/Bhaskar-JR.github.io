---
layout : single  
classes : wide
author_profile: true
comments: true
---

# Declaring global constants
<blockquote class = "green">
<style>
      .my_text
            {
                font-size:      1em;
                font-style: normal;
            }
        </style>
<pre class = "my_text">
Option Base 1

'Declaring Constant variables  
Public Const col1 As String = "A"
Public Const col2 As String = "N"
Public Const col3 As String = "P"

'Assigning initial value as 2, because data transfer will happen from 2nd row onwards
Public Const Starting_row As Integer = 2
Public Const header_row As Integer = 1  

'go to path : '_sass/minimal-mistakes/_base.scss' to modify the blockquote settings
</pre>
</blockquote>


# Module for splitting data and trasnfering to new sheets

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
Sub A1SplitData_NewSheet()

Dim answer As Integer

answer = MsgBox("Are you sure if you want results in new sheets", vbQuestion + vbYesNo + vbDefaultButton2, "Split Data to New Sheets")

If answer = vbNo Then
  Exit Sub
End If

<code style="color:lightgreen;">'to declare variable of worksheet type for main sheet, that has data to split</code>
Dim source_sheet As Worksheet

'to declare variable of worksheet type for adding required sheets
Dim destination_sheet As Worksheet
Dim source_row As Long
Dim last_row As Long
Dim destination_last_row As Long
Dim destination_row As Long

'this variable is for changing values in column O, that has SAG
Dim CI As String

Application.ScreenUpdating = False

'assigning active sheet, that has data to split
Set source_sheet = ActiveSheet
'to know the last filled row and activesheet bases on column O, that had data to split
'last_row = source_sheet.Cells(source_sheet.Rows.Count, col3).End(xlUp).Row  
'Not reliable as it does not enter inside a table list object even if the cell is empty

last_row = source_sheet.ListObjects("Table1").ListColumns(1).DataBodyRange.Find(What:="*", LookIn:=xlValues, SearchOrder:=xlByRows, SearchDirection:=xlPrevious).row

Dim UniqueCI As Variant
'UniqueCI = UniqueVals(col3, source_sheet.Name)
Dim rng_ci As Range

With source_sheet
  Set rng_ci = .Range(.Cells(Starting_row, col3), .Cells(last_row, col3))
  'Note col3 is public ocntant set by the user
End With

<code style="color:yellow;">UniqueCI = getUniques(rng_ci, False)</code>


Dim item As Variant
Dim map_CI_to_sheet_1() As Variant  'Instantiating Array container
Dim map_CI_to_sheet_2() As Variant  'Instantiating Array container
Dim c As Long
c = 1

For Each item In UniqueCI
 'only if there is nonempty item, create new sheet
 If item = "" Then
 Else
    ReDim Preserve map_CI_to_sheet_1(c)
    ReDim Preserve map_CI_to_sheet_2(c)
    map_CI_to_sheet_1(c) = CStr(item)
    <code style="color:yellow;">map_CI_to_sheet_2(c) = Addnewsheet(CStr(item), source_sheet.Name)</code>
    c = c + 1
End If
Next item

    Dim pos As Integer
    ReDim map_CI_to_sheet(UBound(map_CI_to_sheet_1), 2)
    For pos = 1 To UBound(map_CI_to_sheet_1)
        map_CI_to_sheet(pos, 1) = map_CI_to_sheet_1(pos)
        map_CI_to_sheet(pos, 2) = map_CI_to_sheet_2(pos)
    Next pos

For source_row = Starting_row To last_row
    CI = source_sheet.Cells(source_row, col3).Value
    If CStr(CI) = "" Then
      MsgBox "There is empty value in CI column. Ensure non empty values"
    Exit Sub
    End If
    Set destination_sheet = Nothing

    'On Error Resume Next
    Set destination_sheet = Worksheets(CStr(Application.VLookup(CI, map_CI_to_sheet, 2, False)))
    'On Error GoTo 0

'    If destination_sheet Is Nothing Then 'correct
'            Set destination_sheet = Worksheets.Add(After:=Worksheets(Worksheets.Count))
'            'To assign name to added sheet
'            destination_sheet.Name = CI
'            'To add header row to each added sheet

'            source_sheet.Rows(header_row).Copy destination:=destination_sheet.Rows(header_row)
'            source_sheet.Rows(source_row).Copy destination:=destination_sheet.Rows(2)
'
'   Else
            Dim Arr1() As Variant
            With destination_sheet
                 destination_last_row = WorksheetFunction.Max(.Cells(destination_sheet.Rows.Count, col3).End(xlUp).row, 2)
                 Dim RR As Range
                 Set RR = .Range(.Cells(Starting_row, col1), .Cells(destination_last_row, col1))
            End With

            <code style="color:yellow;">arr = CreateArr(RR, destination_last_row)</code>

            Dim x As Boolean
            x = False

            <code style="color:yellow;">Call Compare(source_sheet, source_row, arr, x, col1, col2, col3)</code>

            If x = False Then
            destination_row = destination_sheet.Cells(destination_sheet.Rows.Count, col3).End(xlUp).row + 1
            source_sheet.Rows(source_row).Copy Destination:=destination_sheet.Rows(destination_row)
            End If

'    End If  'correct
Next source_row

source_sheet.Activate

For Each item In Application.Index(map_CI_to_sheet, , 2)
    Call <code style="color:yellow;">Macro1_Analyse_Sheet(CStr(item))</code>

Next item

source_sheet.Activate

Application.ScreenUpdating = True

MsgBox "Exporting of Data to new sheets completed."
End Sub
</code></pre></div></div>

  
# Functions and Procedures called by the A1SplitData_NewSheet

## Comparing
<blockquote class = 'green'>
Private Sub Compare(ByRef source_ws As Worksheet, ByRef source_row As Long, ByRef Arrayc As Variant, ByRef x As Boolean, col1 As String, col2 As String, col3 As String)
<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
    Dim Compare As String
    With source_ws
      Compare = .Cells(source_row, col1) & "" & .Cells(source_row, col2) & "" & .Cells(source_row, col3)
    End With


   'Dim x As Boolean
    Dim item As Variant
   'loop through the entire array
   For Each item In Arrayc
            'show the element in the debug window.
            If StrComp(Compare, item) = 0 Then
              x = True
              Exit For
            Else
            x = False
            End If
    Next item
    Erase Arrayc

</code></pre></div></div>
End Sub
</blockquote>


<blockquote class = 'green'>
Private Function CreateArr(RR As Range, destination_last_row As Long) As Variant()

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
    If RR.Cells.Count = 1 Then 'correct
        ReDim Arr1(1 To 1, 1 To 1)
        Arr1(1, 1) = RR.Value

        ReDim arr2(1 To 1, 1 To 1)
        ReDim Arr3(1 To 1, 1 To 1)

        With Worksheets(RR.Worksheet.Name)

        arr2(1, 1) = .Range(.Cells(Starting_row, col2), .Cells(destination_last_row, col2))
        Arr3(1, 1) = .Range(.Cells(Starting_row, col3), .Cells(destination_last_row, col3))

        End With

        ReDim CreateArr_1(1 To 1, 1 To 1)
        CreateArr_1(1, 1) = Arr1(1, 1) & arr2(1, 1) & Arr3(1, 1)

        CreateArr = CreateArr_1
    Else

    Arr1 = RR

    ReDim arr2(UBound(Arr1), 1)
    ReDim Arr3(UBound(Arr1), 1)

    With Worksheets(RR.Worksheet.Name)
      Dim rng2 As Range, rng3 As Range
      Set rng2 = .Range(.Cells(Starting_row, col2), .Cells(destination_last_row, col2))
      Set rng3 = .Range(.Cells(Starting_row, col3), .Cells(destination_last_row, col3))
      arr2 = rng2
      Arr3 = rng3
    End With

    ReDim CreateArr_1(UBound(Arr1), 1)
    Dim pos As Integer
    '=CONCAT(CHAR(CODE(A17)+1),1)

    For pos = 1 To UBound(Arr1)
        <code style="color:yellow;">CreateArr_1(pos, 1) = Arr1(pos, 1) & arr2(pos, 1) & Arr3(pos, 1)</code>
        'When you bring in data from a worksheet to a VBA array, the array is always 2 dimensional.
        'The first dimension is the rows and the second dimension is the columns.
        'So, in the example above, Arr is implicitly sized as Arr(1 To 5, 1 To 3) where 5 is the number of rows and 3 is the number of columns.
        'A 2 dimensional array is created even if the worksheet data is in a single row or a single column (e.g, Arr(1 To 10, 1 To 1)).
    Next pos
    'destination_row = destination_sheet.Cells(destination_sheet.Rows.Count, col3).End(xlUp).Row + 1
    'source_sheet.Rows(source_row).Copy Destination:=destination_sheet.Rows(destination_row)

    CreateArr = CreateArr_1
    End If
</code></pre></div></div>  
End Function
</blockquote>

<blockquote class = 'green'>
<a id='Addnewsheet'></a>
Private Function Addnewsheet(str As String, source_shtname As String)  
'Adds a new sheet and returns the name of the new sheet

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
Dim Sht As Worksheet, newShtName As String
'Using an array to store preexisting suffixes to newShtName if sheets exist with similar names
Dim arr() As Variant
Dim z As Boolean



Set NewSht = Sheets.Add(After:=ActiveSheet)
newShtName = str
Worksheets(source_shtname).Rows(header_row).Copy Destination:=NewSht.Rows(header_row)

'if "Ops" sheet exists, there will be another added, e.g. "Ops_2"
Dim cnt As Long
cnt = 0
For Each Sht In ActiveWorkbook.Sheets
    If InStr(1, Sht.Name, newShtName) = 1 Then

'Using cnt as counter to know the number of prexisting sheets whose initial letters match newShtName
    cnt = cnt + 1

'Redimming with preserve to only add values using new incremental upper bound
'Note using preserve is important otherwise Arr will reinitialise with empty values
    ReDim Preserve arr(cnt)

'Populating arr with suffixes of sheets whose names have starting letters matching with newShtName
    arr(cnt) = Right(Sht.Name, Len(Sht.Name) - Len(newShtName))
    End If
Next Sht

Do While Not z
    z = False
'Checking if cnt can be used as suffix to newShtName
'For instance, there might be two prexisting sheets named "NewSheet" and "NewSheet_2".
'cnt will be 2 but cannot be used for naming resulting in an error object variable already set

    If IsInArray("_" & CStr(cnt), arr) = False Then
        NewSht.Name = newShtName & IIf(cnt > 0, "_" & cnt, "")
        Addnewsheet = NewSht.Name
        z = True
    Else
        cnt = cnt + 1
        z = False
    End If
Loop
</code></pre></div></div>
End Function  
</blockquote>

<blockquote class = 'green'>
Private Function UniqueVals(col As Variant, Optional SheetName As String = "") As Variant  
    'Return a 1-based array of the unique values in column Col
<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
    Dim D As Variant, a As Variant, v As Variant
    Dim i As Long, n As Long, k As Long
    Dim ws As Worksheet

    If Len(SheetName) = 0 Then
        Set ws = ActiveSheet
    Else
        Set ws = Sheets(SheetName)
    End If

    n = ws.Cells(Rows.Count, col).End(xlUp).row
    ReDim a(1 To n)
    Set D = CreateObject("Scripting.Dictionary")

    For i = 1 To n
        v = ws.Cells(i, col).Value
        If Not D.Exists(v) Then
            D.Add v, 0
            k = k + 1
            a(k) = k
        End If
    Next i

    ReDim Preserve a(1 To k)
    UniqueVals = a
</code></pre></div></div>
End Function
</blockquote>

<blockquote class = 'green'>
Function getUniques(a, Optional ZeroBased As Boolean = True)  
['Link on populate unique values into a vba array from excel](https://stackoverflow.com/questions/5890257/populate-unique-values-into-a-vba-array-from-excel)  
<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
Dim tmp: tmp = Application.Transpose(WorksheetFunction.Unique(a))
If ZeroBased Then ReDim Preserve tmp(0 To UBound(tmp) - 1)
getUniques = tmp
</code></pre></div></div>
End Function
</blockquote>


<blockquote class = "green">
Private Function IsInArray(valToBeFound As Variant, arr As Variant) As Boolean  
'DEVELOPER: Ryan Wells ([wellsr.com](https://wellsr.com))  
'DESCRIPTION: Function to check if a value is in an array of values  
'INPUT: Pass the function a value to search for and an array of values of any data type.  
'OUTPUT: True if is in array, false otherwise   

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>
Dim element As Variant
On Error GoTo IsInArrayError: 'array is empty
    For Each element In arr
        If element = valToBeFound Then
            IsInArray = True
            Exit Function
        End If
    Next element
Exit Function
IsInArrayError:

On Error GoTo 0
IsInArray = False    

</code></pre></div></div>
End Function
</blockquote>
