# Analysis of an Evolutionary Algorithm for Crossword Puzzle Generation
**Author:** Mohammad Safi  
**Date:** December 11, 2023

## Introduction

Evolutionary Algorithms (EAs) are a subset of evolutionary computation inspired by biological evolution. This project employs an EA for generating crossword puzzles, given their vast solution space and the need for satisfying various conditions.

## Description of the Evolutionary Algorithm

The EA used in this project evolves a population of potential crossword solutions towards an optimal configuration, guided by a defined fitness function. It emulates natural selection, with only the fittest individuals chosen for reproduction.

### Population Initialization

1. The longest word is placed in the middle of the grid.
2. Words with common letters with the longest word are placed intersecting correctly.
3. Words without common letters are placed randomly.

### Fitness Evaluation

Fitness is evaluated based on:
- Correct intersections of words.
- Proper placement without two words of the same orientation next to each other.
- No parallel words forming new words.
- All words being connected.
- No missing or duplicated words.
- No invalid words formed in the grid.

### Selection, Crossover, Mutation, and Replacement

- **Selection:** Fittest crosswords are selected for reproduction.
- **Crossover:** Combines features from two parents to create a new child.
- **Mutation:** Introduces changes in the child crosswords.
- **Replacement:** Older generation is replaced with the new one.

### Termination

The algorithm runs for a predefined number of generations or until a desired fitness level is achieved.

## Variation Operators and EA Parameters

- **Crossover Operator:** Uses a midpoint strategy for inheritance and diversity.
- **Mutation Operator:** Repositions words with the highest penalties.
- **EA Parameters:** Population size, maximum generations, desired fitness level.

## Data Collection for Report

### Methodology for Running the Tests

1. Test Preparation: Set up input files with different word lists.
2. Initialization: Parameters like population size and max generations are set.
3. Execution of Tests: Run the EA for each word list.
4. Data Collection: Collect data on word lists, puzzles, fitness, and duration.
5. Output Generation: Store word placements and grid layouts.
6. Statistical Analysis: Record statistics for each test.

### Statistical Data

## Statistical Data Summary

The following table summarizes the results from the tests:


| Test No. | Word Count | Avg Fitness | Max Fitness | Duration (s) |
|----------|------------|-------------|-------------|--------------|
| 1        | 10         | 10.99       | 13          | 93.13        |
| 2        | 8          | 2.43        | 4           | 116.12       |
| 3        | 10         | 4.28        | 3           | 140.22       |
| 4        | 10         | 12.59       | 18          | 74.91        |
| 5        | 6          | 3.28        | 5           | 95.82        |
| 6        | 9          | 11.45       | 8           | 124.68       |
| 7        | 10         | 5.77        | 11          | 153.39       |
| 8        | 10         | 11.35       | 15          | 123.75       |
| 9        | 7          | 7.00        | 7           | 51.29        |
| 10       | 5          | 7.28        | 8           | 29.63        |
| 11       | 6          | 8.05        | 4           | 86.42        |
| 12       | 9          | 10.51       | 16          | 63.96        |
| 13       | 10         | 6.16        | 2           | 127.25       |
| 14       | 9          | 0.31        | 4           | 2832.20      |
| 15       | 7          | 7.40        | 12          | 793.14       |
| 16       | 7          | 4.63        | 8           | 111.35       |
| 17       | 8          | 6.35        | 9           | 119.23       |
| 18       | 9          | 2.16        | 0           | 134.33       |
| 19       | 6          | 8.27        | 10          | 98.33        |
| 20       | 9          | 4.01        | 8           | 144.06       |
| 21       | 6          | 7.52        | 4           | 79.79        |
| 22       | 5          | 3.33        | 8           | 87.84        |
| 23       | 7          | 1.75        | 1           | 101.77       |
| 24       | 7          | 6.03        | 7           | 117.64       |
| 25       | 7          | 8.97        | 5           | 101.65       |
| 26       | 5          | 4.00        | 8           | 91.43        |
| 27       | 9          | 4.27        | 9           | 123.34       |
| 28       | 7          | 8.55        | 4           | 95.87        |
| 29       | 9          | 10.65       | 16          | 104.00       |
| 30       | 8          | 6.56        | 4           | 101.46       |
| ...      | ...        | ...         | ...         | ...          |
| 98       | 8          | 4.80        | 3           | 113

## Statistical Analysis and Plot Generation

### Analysis Overview

- Data Categorization: Grouped by word count.
- Average Fitness: Indicates puzzle quality.
- Maximum Fitness: Represents the best solution quality.

### General Observations

- Consistency and challenge in complexity vary with word count.
- Algorithm efficiency and outliers are notable factors.

### Plots

### Average Fitness vs. Word Count and Maximum Fitness vs. Word Count

![Average Fitness vs. Word Count](/images/Figure_1.png)

*Description 1: The plot demonstrates a general increasing trend in average fitness with the word count rising from 5 to 10.*

*Description 2: The maximum fitness scores present a contrasting trend to the average fitness scores.*

### Time to the Number of Words

![Time to the Number of Words](/images/TimeFigure.png)

*Description: The plot shows a steady increase in the average duration for word counts ranging from 5 to 7.*
## Summary

The statistical analysis evaluated the efficiency and performance of the crossword puzzle algorithm. Key findings include variability in average and maximum fitness scores, performance spikes at certain word counts, and potential areas for algorithmic optimization.
