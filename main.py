import os
import random

# List to store original words that exist in the input file
ORIGINAL_WORDS = []


class Word:
    def __init__(self, word, row, column, orientation):
        """
        Represents a word in the crossword with its properties.

        Args:
        word (str): The actual word.
        row (int): The starting row of the word in the grid.
        column (int): The starting column of the word in the grid.
        orientation (int): The orientation of the word (0 for horizontal, 1 for vertical).
        """
        self.word = word
        self.row = row
        self.column = column
        self.orientation = orientation
        # Penalty score for the word, initialized to 0 (Used to know which word is causing problems).
        self.penalty = 0


class Crossword:
    def __init__(self):
        """
        Represents the crossword grid and its associated operations.
        """
        # 20x20 grid initialization with empty strings
        self.grid = [['' for _ in range(20)] for _ in range(20)]
        # List to store the words placed on the grid.
        self.words = []

    def remove_word(self, word_to_remove):
        """
        Removes a word from the grid and the list of words.

        Args:
        word_to_remove (Word): The word object to remove.
        """
        # Clear the cells occupied by the word
        for i in range(len(word_to_remove.word)):
            if word_to_remove.orientation == 0:  # Horizontal
                self.grid[word_to_remove.row][word_to_remove.column + i] = ''
            else:  # Vertical
                self.grid[word_to_remove.row + i][word_to_remove.column] = ''

        # Remove the word object from the list of words
        self.words = [word for word in self.words if word != word_to_remove]

    def can_place_word(self, word, row, column, orientation):
        """
        Checks if a word can be placed at a specified position in the grid.

        Args:
        word (str): The word to be placed.
        row (int): Row position for the word.
        column (int): Column position for the word.
        orientation (int): Orientation of the word (0 for horizontal, 1 for vertical).

        Returns:
        bool: True if the word can be placed, False otherwise.
        """
        if orientation == 0:  # Horizontal
            # Check if word goes beyond grid boundaries
            if column + len(word) > 20:
                return False
            # Check for conflicts with existing words
            for i in range(len(word)):
                if self.grid[row][column + i] != '' and self.grid[row][column + i] != word[i]:
                    return False
        else:  # Vertical
            if row + len(word) > 20:
                return False
            for i in range(len(word)):
                if self.grid[row + i][column] != '' and self.grid[row + i][column] != word[i]:
                    return False
        return True

    def add_word(self, word):
        """
        Adds a word to the grid if it fits.

        Args:
        word (Word): The word object to add.

        Returns:
        bool: True if the word is successfully added, False otherwise.
        """
        # Check if word fits in the grid for the given orientation
        if word.orientation == 0 and word.column + len(word.word) > 20:
            return False  # Word does not fit horizontally
        elif word.orientation == 1 and word.row + len(word.word) > 20:
            return False  # Word does not fit vertically
        # Place the word in the grid
        for i in range(len(word.word)):
            if word.orientation == 0:  # Horizontal
                self.grid[word.row][word.column + i] = word.word[i]
            else:  # Vertical
                self.grid[word.row + i][word.column] = word.word[i]

        # Add the word to the words list
        self.words.append(word)
        return True

    def print_grid(self):
        """
        Prints the current state of the crossword grid.
        """
        for row in self.grid:
            for cell in row:
                # Print each cell or a dot if empty
                print(cell if cell else '.', end=' ')
            print()  # Newline after each row

    def no_intersecting_same_orientation(self):
        """
        Calculates a fitness penalty for words in the same orientation intersecting each other.

        Returns:
        int: The calculated fitness penalty.
        """
        fitness = 0
        for word in self.words:
            row, col = word.row, word.column
            # Check each character in the word
            for i in range(len(word.word)):
                # Check for intersection conflicts for vertical words
                if word.orientation == 1:  # Vertical
                    # Check the cell below the last character and the cell above the first character
                    if row < 19 and self.grid[row + 1][col] != '' and i == len(word.word) - 1:
                        fitness -= 1
                        word.penalty += 1
                    elif row > 0 and self.grid[row - 1][col] != '' and i == 0:
                        fitness -= 1
                        word.penalty += 1
                    row += 1
                else:  # Horizontal
                    # Check the cell to the right of the last character and the cell to the left of the first character
                    if col < 19 and self.grid[row][col + 1] != '' and i == len(word.word) - 1:
                        fitness -= 1
                        word.penalty += 1
                    elif col > 0 and self.grid[row][col - 1] != '' and i == 0:
                        fitness -= 1
                        word.penalty += 1
                    col += 1
        return fitness

    def is_valid_word(self, word):
        """
        Checks if a word is in the list of original words.

        Args:
        word (str): The word to check.

        Returns:
        bool: True if the word is valid, False otherwise.
        """
        global ORIGINAL_WORDS
        return word in ORIGINAL_WORDS

    def form_perpendicular_word(self, row, col, orientation):
        """
        Forms a word perpendicular to the specified orientation starting from the given cell.
        Used in the function each_word_crossed_correctly to make sure there is no problem when the words are crossing.

        Args:
        row (int): The row index of the starting cell.
        col (int): The column index of the starting cell.
        orientation (int): The orientation of the word to form perpendicular to (0 for horizontal, 1 for vertical).

        Returns:
        str: The formed perpendicular word.
        """
        word = ''
        if orientation == 0:  # Horizontal word, form vertical
            # Move upwards to start of potential word
            while row > 0 and self.grid[row - 1][col] != '':
                row -= 1
            # Form the word moving downwards
            while row < 20 and self.grid[row][col] != '':
                word += self.grid[row][col]
                row += 1
        else:  # Vertical word, form horizontal
            # Move leftwards to start of potential word
            while col > 0 and self.grid[row][col - 1] != '':
                col -= 1
            # Form the word moving rightwards
            while col < 20 and self.grid[row][col] != '':
                word += self.grid[row][col]
                col += 1
        return word

    def check_grid_for_invalid_words(self):
        """
        Scans the grid horizontally and vertically for invalid words (words not in the original list).

        Returns:
        int: The total fitness penalty for all invalid words found in the grid.
        """
        fitness = 0

        # Horizontal check
        for row in range(20):  # Assuming grid size is 20x20
            potential_word = ''
            for col in range(20):
                if self.grid[row][col] != '':  # Non-empty cell
                    potential_word += self.grid[row][col]
                else:  # Empty cell or end of row
                    if len(potential_word) > 1 and not self.is_valid_word(potential_word):
                        fitness -= 1  # Penalize for invalid word
                    potential_word = ''  # Reset potential word

            # Check at the end of the row if the last cell was not empty
            if len(potential_word) > 1 and not self.is_valid_word(potential_word):
                fitness -= 1

        # Vertical check
        for col in range(20):
            potential_word = ''
            for row in range(20):
                if self.grid[row][col] != '':
                    potential_word += self.grid[row][col]
                else:
                    # Check for invalid word
                    if len(potential_word) > 1 and not self.is_valid_word(potential_word):
                        fitness -= 1
                    potential_word = ''

            # Check at the end of the column if the last cell was not empty
            if len(potential_word) > 1 and not self.is_valid_word(potential_word):
                fitness -= 1

        return fitness

    def each_word_crossed_correctly(self):
        """
        Evaluates the fitness by checking if each word in the grid intersects correctly with others.

        Returns:
        int: The fitness score based on correct intersections.
        """
        fitness = 0
        for word in self.words:
            for i in range(len(word.word)):
                row, col = word.row, word.column
                if word.orientation == 0:  # Horizontal
                    col += i
                else:  # Vertical
                    row += i

                # Check if the word intersects correctly
                if self.is_intersecting(row, col, word.orientation):
                    intersecting_char = self.grid[row][col]
                    if intersecting_char == word.word[i]:
                        # Form a word in the perpendicular direction
                        perpendicular_word = self.form_perpendicular_word(
                            row, col, word.orientation)
                        if self.is_valid_word(perpendicular_word):
                            fitness += 1  # Reward for correct intersection

                        else:
                            fitness -= 1  # Penalty for incorrect intersection
                            word.penalty += 1
                    else:
                        fitness -= 1  # Penalty for incorrect intersection
                        word.penalty += 1

        return fitness

    def is_intersecting(self, row, col, orientation):
        """
        Checks if a cell at a given position intersects with a word of a different orientation.
        Used in the function each_word_crossed_correctly .

        Args:
        row (int): Row index of the cell.
        col (int): Column index of the cell.
        orientation (int): Orientation of the word being checked.

        Returns:
        bool: True if there is an intersection, False otherwise.
        """
        # Check if the cell at (row, col) intersects with a word of a different orientation
        if orientation == 0:  # Horizontal word
            # Check the cell directly above and below
            if row > 0 and self.grid[row - 1][col] != '':
                return True
            if row < 19 and self.grid[row + 1][col] != '':
                return True
        else:  # Vertical word
            # Check the cell directly to the left and right
            if col > 0 and self.grid[row][col - 1] != '':
                return True
            if col < 19 and self.grid[row][col + 1] != '':
                return True
        return False

    def no_parallel_words_forming_new_words(self):
        """
        Calculates a fitness penalty for parallel words that accidentally form new words.

        Returns:
        int: The fitness penalty for parallel words forming new words.
        """

        fitness = 0

        for word in self.words:  # Iterate excluding the last character
            for i in range(len(word.word) - 1):  # Exclude the last character
                row, col = word.row, word.column
                # Adjust row and column based on the word's orientation
                if word.orientation == 0:  # Horizontal word
                    col += i
                    # Check above and below for both current and next character
                    if (row > 0 and self.grid[row - 1][col] != '' and self.grid[row - 1][col + 1] != '') or \
                            (row < 19 and self.grid[row + 1][col] != '' and self.grid[row + 1][col + 1] != ''):
                        fitness -= 1  # Penalty for parallel word
                        word.penalty += 1

                else:  # Vertical word
                    row += i
                    # Check left and right for both current and next character
                    if (col > 0 and self.grid[row][col - 1] != '' and self.grid[row + 1][col - 1] != '') or \
                            (col < 19 and self.grid[row][col + 1] != '' and self.grid[row + 1][col + 1] != ''):
                        fitness -= 1  # Penalty for parallel word
                        word.penalty += 1

        return fitness

    def is_word_connected(self, word, visited):
        """
        Helper Function Checks if a word is connected to the main graph of words in the crossword.

        Args:
        word (Word): The word to be checked.
        visited (set): Set of visited cells.

        Returns:
        bool: True if the word is connected, False otherwise.
        """
        for i in range(len(word.word)):
            row = word.row + i if word.orientation == 1 else word.row
            col = word.column + i if word.orientation == 0 else word.column
            if (row, col) in visited:
                return True
        return False

    def all_words_connected(self):
        """
        Evaluates whether all words in the crossword are connected.

        Returns:
        int: Fitness score based on the connectivity of words.
        """
        fitness = 0  # Start with no penalty

        # Find the first non-empty cell
        start_row, start_col = next(((r, c) for r in range(
            20) for c in range(20) if self.grid[r][c] != ''), (None, None))

        # If no non-empty cell is found, apply a penalty for having no words
        if start_row is None:
            return -100  # Penalty for an empty grid

        # Run DFS to find all connected cells
        visited = set()
        self.dfs(start_row, start_col, visited)

        # Check each word for connection
        for word in self.words:
            if not self.is_word_connected(word, visited):
                fitness -= 10  # Penalty for each unconnected word

        return fitness

    def dfs(self, row, col, visited):
        """
        Depth-First Search to visit all connected cells in the crossword grid.

        Args:
        row (int): The row index from where to start the search.
        col (int): The column index from where to start the search.
        visited (set): Set to keep track of visited cells.
        """
        if (row, col) in visited or not (0 <= row < 20 and 0 <= col < 20) or self.grid[row][col] == '':
            return
        visited.add((row, col))
        # Possible directions to move
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            self.dfs(row + dr, col + dc, visited)

    def check_missing_or_duplicated_words(crossword):
        """
        Checks for missing or duplicated words in the crossword.

        Args:
        crossword (Crossword): The crossword object to check.

        Returns:
        int: Fitness penalty based on missing or duplicated words.
        """
        global ORIGINAL_WORDS
        fitness = 0
        word_count = {}
        original_words = ORIGINAL_WORDS

        # Count the occurrences of each word in the crossword
        for word in crossword.words:
            word_str = word.word
            word_count[word_str] = word_count.get(word_str, 0) + 1

        # Check for missing and duplicated words
        for original_word in original_words:
            count = word_count.get(original_word, 0)
            if count == 0:
                fitness -= 5  # Penalize for missing word
            elif count > 1:
                fitness -= 5 * (count - 1)  # Penalize for duplicated words

        return fitness

    def calculateFitness(self):
        """
        Calculates the overall fitness score for the crossword.

        Returns:
        int: The total fitness score.
        """
        fitness = 0

        # Points for each word crossed by another word correctly.
        fitness += self.each_word_crossed_correctly()

        # Find if any words have the same orintation and there is no space between them.
        fitness += self.no_intersecting_same_orientation()

        # Subtract points for parallel words forming new words.
        fitness += self.no_parallel_words_forming_new_words()

        # Points for all words being connected.
        fitness += self.all_words_connected()

        # check for missing or dublicated words.
        fitness += self.check_missing_or_duplicated_words()

        # Walk through the grid and check if there is any incorrect words being formed.
        fitness += self.check_grid_for_invalid_words()

        return fitness


def createInitialPopulation(words, populationSize):
    """
    Creates an initial population of crossword puzzles.

    Args:
    words (list): A list of words to be used in the crosswords.
    populationSize (int): The size of the population to generate.

    Returns:
    list: A list of Crossword objects representing the initial population.
    """
    population = []
    for _ in range(populationSize):
        crossword = createRandomCrossword(words)
        population.append(crossword)
    return population


def createRandomCrossword(words):
    """
    Creates a random crossword puzzle using the given words but with some modifcation to create a good population.

    Args:
    words (list): A list of words to be used in the crossword.

    Returns:
    Crossword: A randomly generated crossword object.
    """
    crossword = Crossword()
    words_sorted = sorted(words, key=len, reverse=True)  # Sort words by length
    longest_word = words_sorted[0]
    placed_words = [longest_word]  # List to keep track of placed words

    # Determine center position
    center_row, center_col = 10, 10
    # Randomly choose horizontal or vertical
    orientation = random.choice([0, 1])

    # Place the longest word
    if orientation == 0:  # Horizontal
        start_col = center_col - len(longest_word) // 2
        longest_word_obj = Word(
            longest_word, center_row, start_col, orientation)
    else:  # Vertical
        start_row = center_row - len(longest_word) // 2
        longest_word_obj = Word(longest_word, start_row,
                                center_col, orientation)

    crossword.add_word(longest_word_obj)

    # Try to add intersecting words
    for i in range(len(longest_word)):
        letter = longest_word[i]
        for word in words_sorted:
            if word[0] == letter and word not in placed_words:
                if orientation == 0:  # Longest word horizontal, intersect vertically
                    row = center_row + i
                    col = start_col
                    new_word_obj = Word(word, row, col, 1)
                else:  # Longest word vertical, intersect horizontally
                    row = start_row + i
                    col = center_col
                    new_word_obj = Word(word, row, col, 0)

                if crossword.can_place_word(new_word_obj.word, new_word_obj.row, new_word_obj.column, new_word_obj.orientation):
                    crossword.add_word(new_word_obj)
                    placed_words.append(word)
                    # break

    # Place remaining words randomly but far from the center
    for word in words_sorted:
        if word not in placed_words:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                new_row, new_col = random_position_far_from_center()
                new_orientation = random.randint(0, 1)
                new_word_obj = Word(word, new_row, new_col, new_orientation)

                if crossword.can_place_word(new_word_obj.word, new_row, new_col, new_orientation):
                    crossword.add_word(new_word_obj)
                    placed_words.append(word)
                    placed = True
                attempts += 1

    return crossword


def random_position_far_from_center():
    """
    Generates a random position far from the center of a 20x20 grid.

    Returns:
    tuple: A tuple (row, col) representing the random position.
    """
    # Define 'far away' regions (outer 5 rows/columns in a 20x20 grid)
    far_positions = list(range(0, 5)) + list(range(15, 20))
    row = random.choice(far_positions)
    col = random.choice(far_positions)
    return row, col


def calculateFitness(crossword):
    """
    Calculates the fitness of a given crossword puzzle.

    Args:
    crossword (Crossword): The crossword puzzle to evaluate.

    Returns:
    int: The fitness score of the crossword.
    """

    fitness = crossword.calculateFitness()

    return fitness


def selectParents(population):
    """
    Selects the fittest half of the population to be parents for the next generation.

    Args:
    population (list): The current population of crosswords.

    Returns:
    list: A list of Crossword objects selected as parents.
    """

    population.sort(key=lambda x: calculateFitness(x), reverse=True)
    return population[:len(population)//2]


def crossover(parent1, parent2):
    """
    Performs a crossover operation between two parent crosswords to create a child.

    Args:
    parent1 (Crossword): The first parent crossword.
    parent2 (Crossword): The second parent crossword.

    Returns:
    Crossword: The resulting child crossword after the crossover.
    """
    child = Crossword()

    # Splitting the words from both parents at midpoint
    mid_point = len(parent1.words) // 2
    child_words = parent1.words[:mid_point] + parent2.words[mid_point:]

    # Adding words to the child's crossword
    for word in child_words:
        child.add_word(word)

    return child


def mutate(crossword):
    """
    Randomly mutates a word's position in the given crossword.

    Args:
    crossword (Crossword): The crossword to mutate.
    """
    if not crossword.words:
        return  # No mutation if there are no words

    # Choose a random word to mutate
    word_to_mutate = random.choice(crossword.words)

    attempts = 0
    max_attempts = 50  # Max attempts to find a new position

    while attempts < max_attempts:
        # Generate new position and orientation
        new_row = random.randint(0, 19)
        new_column = random.randint(0, 19)
        new_orientation = random.randint(0, 1)
        mutated_word = Word(word_to_mutate.word, new_row,
                            new_column, new_orientation)

        # Check if the mutated word can be placed in the new position
        if crossword.can_place_word(mutated_word.word, new_row, new_column, new_orientation):
            # Remove the original word and add the mutated word
            crossword.remove_word(word_to_mutate)
            crossword.add_word(mutated_word)
            return  # Successful mutation

        attempts += 1


def reset_penalties(crossword):
    """
    Resets the penalty scores for all words in the crossword.

    Args:
    crossword (Crossword): The crossword whose word penalties are to be reset.
    """
    for word in crossword.words:
        word.penalty = 0


def replacePopulation(population, offspring):
    """
    Replaces the current population with the offspring, keeping the population size constant.

    Args:
    population (list): The current population.
    offspring (list): The newly generated offspring.

    Returns:
    list: The new population consisting of the fittest individuals from both the current population and the offspring.
    """
    combined = population + offspring
    combined.sort(key=lambda x: calculateFitness(x), reverse=True)
    return combined[:len(population)]


def runEvolutionaryAlgorithm(words, populationSize, maxGenerations, desiredFitness):
    """
    Runs the evolutionary algorithm to find the best crossword layout.

    Args:
    words (list): The list of words to use in the crossword.
    populationSize (int): The size of the population in each generation.
    maxGenerations (int): The maximum number of generations to run the algorithm.
    desiredFitness (int): The desired fitness level to achieve.

    Returns:
    Crossword: The crossword with the highest fitness score.
    """
    # Create an initial population of crossword puzzles.
    population = createInitialPopulation(words, populationSize)

    # Iterate over the specified number of generations.
    for _ in range(maxGenerations):
        # Evaluate and print the fitness of each crossword in the current population.
        for crossword in population:
            print("The current Fitness is: ", crossword.calculateFitness())
            fitness = crossword.calculateFitness()

            # If a crossword's fitness meets or exceeds the desired fitness, return it as the solution.
            if fitness >= desiredFitness:
                return crossword

        # Select the fittest half of the population as parents for the next generation.
        parents = selectParents(population)
        offspring = []

        # Create offspring from the selected parents through crossover and mutation.
        for i in range(len(parents)//2):
            # Create a child crossword from two parents.
            child = crossover(parents[i], parents[len(parents) - i - 1])
            # Apply mutation to introduce genetic variation.
            mutate(child)
            offspring.append(child)

        # Replace the current population with the offspring, maintaining the population size.
        population = replacePopulation(population, offspring)

        # Reset penalty scores for each crossword in the population.
        for crossword in population:
            reset_penalties(crossword)

    # If the desired fitness level is not achieved within the maximum generations,
    # return the crossword with the highest fitness from the final population.
    return max(population, key=lambda x: calculateFitness(x))


def generateOutput(bestCrossword, original_words):
    """
    Generates the output of the best crossword solution.

    Args:
    bestCrossword (Crossword): The best crossword puzzle.
    original_words (list): The list of original words used in the crossword.

    Returns:
    list: A list of strings representing the placement of each word in the crossword.
    """
    word_placement_dict = {word.word: (
        word.row, word.column, word.orientation) for word in bestCrossword.words}

    output = []
    for word in original_words:
        if word in word_placement_dict:
            placement = word_placement_dict[word]
            # Format: row, column, orientation
            output.append(f"{placement[0]} {placement[1]} {placement[2]}")
        else:
            # Handle the case where a word is not found
            output.append("Word not found")

    return output


def read_words_from_file(file_path):
    """
    Reads words from a given file.

    Args:
    file_path (str): The path to the file containing words.

    Returns:
    list: A list of words read from the file.
    """
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words


def read_input_files(directory_path):
    """
    Reads all word files from a given directory.

    Args:
    directory_path (str): The path to the directory containing word files.

    Returns:
    list: A list containing lists of words from each file.
    """
    all_words = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            words = read_words_from_file(file_path)
            all_words.append(words)
    return all_words


def grid_to_string(grid):
    """
    Converts the crossword grid to a string format.

    Args:
    grid (list): The crossword grid.

    Returns:
    str: The grid in string format.
    """
    grid_str = ''
    for row in grid:
        row_str = ' '.join([cell if cell else '.' for cell in row])
        grid_str += row_str + '\n'
    return grid_str


def main():
    """
    Main function to run the crossword generation algorithm.
    """
    directory_path = './inputs'
    global ORIGINAL_WORDS
    all_input_word_lists = read_input_files(directory_path)
    populationSize = 75  # population size
    maxGenerations = 5000  # number of generations

    # Iterate over each list of words (from each file).
    for idx, word_list in enumerate(all_input_word_lists):
        ORIGINAL_WORDS = word_list
        maxFitness = len(word_list) + 2
        # Set desired fitness (can be lower than max)
        desiredFitness = maxFitness - 2

        best_crossword = runEvolutionaryAlgorithm(
            word_list, populationSize, maxGenerations, desiredFitness)
        output = generateOutput(best_crossword, ORIGINAL_WORDS)

        output_file_path = f'./output{
            idx + 1}.txt'
        with open(output_file_path, 'w') as f:
            for line in output:
                f.write(line + '\n')
            f.write(grid_to_string(best_crossword.grid))


        print("\nBest Crossword Grid for Input", idx + 1)
        best_crossword.print_grid()  # Print the grid of the best crossword


if __name__ == "__main__":
    main()
