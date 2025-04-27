import os
from typing import Callable, List, Tuple
from random import shuffle, choice
from queue import PriorityQueue
from collections import defaultdict
import time
import csv

# HEURISTICS

def misplaced_tiles(state: Tuple[int]) -> int:
    return sum(1 for i, value in enumerate(state) if value != 0 and value != i + 1)

def manhattan_distance(state: Tuple[int]) -> int:
    n = int(len(state) ** 0.5)
    distance = 0
    for index, value in enumerate(state):
        if value == 0:
            continue
        curr_row, curr_col = divmod(index, n)
        goal_row, goal_col = divmod(value - 1, n)
        distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
    return distance

def manhattan_linear_conflict(state: Tuple[int]) -> int:
    n = int(len(state) ** 0.5)
    goal_position = tuple(range(1, n * n)) + (0,)
    manhattan = manhattan_distance(state)
    linear_conflict = 0

    for row in range(n):
        max_goal_col = -1
        for col in range(n):
            index = row * n + col
            value = state[index]
            if value == 0:
                continue
            goal_row, goal_col = goal_position[value]
            if goal_row == row:
                if goal_col > max_goal_col:
                    max_goal_col = goal_col
                else:
                    linear_conflict += 1
    
        for col in range(n):
            max_goal_row = -1
            for row in range(n):
                index = row * n + col
                value = state[index]
                if value == 0:
                    continue
                goal_row, goal_col = goal_position[value]
                if goal_col == col:
                    if goal_row > max_goal_row:
                        max_goal_row = goal_row
                    else:
                        linear_conflict += 1

        return manhattan + 2 * linear_conflict

# PUZZLE

# [1, 2, 3, 4]
# [5, 6, 7, 8]
# [9, 10, 11, 12]
# [13, 14, 15, 0]
def generate_solved_puzzle(n):
    return tuple(range(1, n * n)) + (0,)

def generate_puzzle(n):    
    tiles = list(range(1, n * n))
    shuffle(tiles)
    return tuple(tiles) + (0,) 

def is_solvable(puzzle: Tuple[int]) -> bool:
    n = int(len(puzzle) ** 0.5)
    inv_count = sum(
        1 for i in range(len(puzzle)) for j in range(i + 1, len(puzzle))
        if puzzle[i] and puzzle[j] and puzzle[i] > puzzle[j]
    )
    if len(puzzle) % 2 == 1:
        # nieparzysta -> parzyście inwersji
        return inv_count % 2 == 0
    else:
        # parzysta -> 
        # nieparzyście inwersji + zero na parzystym wierszu od dołu 
        # lub 
        # nieparzyście inwersji + zero na nieparzystym wierszu od dołu
        row_blank = puzzle.index(0) // n
        return (inv_count + row_blank) % 2 == 1
    
def generate_solvable_puzzle(n):
    while True:
        puzzle = generate_puzzle(n)
        if is_solvable(puzzle):
            return puzzle
        
# LOGIC 
     
def moves(puzzle: Tuple[int]) -> list:
    n = int(len(puzzle) ** 0.5)
    idx = puzzle.index(0)
    x, y = divmod(idx, n)
    moves = []
    dirs = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    for move, (dx, dy) in dirs.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            n_idx = nx * n + ny
            new_puzzle = list(puzzle)
            #zamiana miejscami 0 z sąsiadem
            new_puzzle[idx], new_puzzle[n_idx] = new_puzzle[n_idx], new_puzzle[idx]
            # dodajemy nowy stan do listy ruchów
            moves.append((tuple(new_puzzle), str(puzzle[n_idx])))
    return moves

# A* ALGORITHM

def astar(start_puzzle: Tuple[int], heuristic: Callable[[Tuple[int]], int], final_state: Tuple[int]) -> Tuple[List[str], int, int]:
    n = int(len(start_puzzle) ** 0.5)
    open_set = PriorityQueue() # kolejka priorytetowa (f-score, stan)
    open_set.put((0, start_puzzle)) # startowy stan
    came_from = {} # rekonstruowanie ścieżki
    g_score = defaultdict(lambda: float('inf')) # ile kosztuje dojście do danego stanu
    g_score[start_puzzle] = 0
    f_score = defaultdict(lambda: float('inf')) # szacowany koszt dojścia do celu g(n) + h(n)
    f_score[start_puzzle] = heuristic(start_puzzle)
    visited = set()

    while not open_set.empty():
        # najniższy koszt (f-score)
        current = open_set.get()[1]
        if current in visited:
            continue
        visited.add(current)

        # sprawdzamy czy dotarliśmy do celu
        if current == final_state:
            path = []
            while current in came_from:
                path.append(came_from[current])
                current = came_from[current]
            return path[::-1], len(path), len(visited)

        # sprawdzamy sąsiadów
        for neighbor, move in moves(current):
            tentative_g_score = g_score[current] + 1 # koszt dotarcia do sąsiada to koszt dotarcia do current + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                # f(n) = g(n) + h(n)
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                # dodajemy sąsiada do kolejki
                open_set.put((f_score[neighbor], neighbor))

    return [], 0, len(visited)

# PUZZLE CREATED BY X RANDOM MOVES FROM A SOLVED STATE

def generate_x_moves_from_goal(n, k):
    state = generate_solved_puzzle(n)
    previous = None

    for _ in range(k):
        neighbors = moves(state)
        # unikamy cofania się do poprzedniego stanu
        if previous:
            neighbors = [nb for nb in neighbors if nb[0] != previous]

        # jeśli nie ma możliwych ruchów, to kończymy
        if not neighbors:
            break  

        previous = state
        # losowo wybieramy sąsiada
        state = choice(neighbors)[0]

    return state

# ====================== TESTS  ======================

def save_results_to_csv(filename: str, results: List[dict]):
    if not results:
        print("Brak wyników do zapisania.")
        return

    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

# ========================== TEST 1 ==========================

def compare_steps_visited_and_time(n: int, trials: int = 50):
    results = []
    solved_puzzle = generate_solved_puzzle(n)

    heuristics = [
        (manhattan_distance, "Manhattan"),
        (misplaced_tiles, "Misplaced"),
        (manhattan_linear_conflict, "ManhattanLinear")
    ]

    for _ in range(trials):
        print(f"\nTrial {_ + 1}/{trials}")
        
        #start_state = generate_x_moves_from_goal(n, 30)
        start_state = generate_solvable_puzzle(n)
        for heuristic_func, heuristic_name in heuristics:
            start_time = time.time()
            path, steps, visited = astar(start_state, heuristic_func, solved_puzzle)
            elapsed_time = round((time.time() - start_time), 2)
            results.append({
                "heuristic": heuristic_name,
                "steps": steps,
                "visited": visited,
                "time": elapsed_time
            })

    save_results_to_csv("data/steps_vs_visited3.csv", results)
    print("Zapisano steps_vs_visited3.csv")

# ========================== TEST 2 ==========================

def compare_heuristics(n: int, k: int, puzzles: int = 10, runs: int = 10):
    results = []
    solved = generate_solved_puzzle(n)
    
    for puzzle_num in range(puzzles):  
        print(f"\nPuzzle {puzzle_num + 1}/{puzzles}")
        #puzzle = generate_x_moves_from_goal(n, k)
        puzzle = generate_solvable_puzzle(n)
        for i in range(runs):
            print(f"Run {i + 1}/{runs}")
           
            path_manhattan, steps_manhattan, visited_manhattan = astar(puzzle, manhattan_distance, solved)
            path_manhattan_linear, steps_manhattan_linear, visited_manhattan_linear = astar(puzzle, manhattan_linear_conflict, solved)
            path_misplaced, steps_misplaced, visited_misplaced = astar(puzzle, misplaced_tiles, solved)

            results.append({
                "puzzle": puzzle_num + 1,
                "run": i + 1,
                "steps_manhattan": steps_manhattan,
                "visited_manhattan": visited_manhattan,
                "steps_manhattan_linear": steps_manhattan_linear,
                "visited_manhattan_linear": visited_manhattan_linear,
                "steps_misplaced": steps_misplaced,
                "visited_misplaced": visited_misplaced
            })

    return results

def summarize_results(results: List[dict]):
    def avg(key):
        return sum(r[key] for r in results) / len(results)

    print("\n--- AVERAGE RESULTS ---")
    print(f"Manhattan: avg steps = {avg('steps_manhattan'):.2f}, avg visited = {avg('visited_manhattan'):.2f}")
    print(f"ManhattanLinear: avg steps = {avg('steps_manhattan_linear'):.2f}, avg visited = {avg('visited_manhattan_linear'):.2f}")
    print(f"Misplaced: avg steps = {avg('steps_misplaced'):.2f}, avg visited = {avg('visited_misplaced'):.2f}")

def run_heuristic_comparison(n: int):
    k = 10
    runs = 10
    results = compare_heuristics(n, k, runs)
    summarize_results(results)
    save_results_to_csv("data/heuristics_comparison.csv", results)
    print("Zapisano heuristics_comparison.csv")

def normal_test(n: int):
    print("Generating a solved puzzle...")
    solved_puzzle = generate_solved_puzzle(n)
    print("Solved puzzle:", solved_puzzle)

    print("\nGenerating a solvable puzzle...")
    puzzle = generate_solvable_puzzle(n)
    #puzzle = (15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0)
    #puzzle = (6, 8, 3, 5, 13, 14, 15, 2, 1, 10, 9, 11, 12, 4, 7, 0)
    #puzzle = (12, 1, 10, 2, 7, 11, 4, 14, 5, 9, 15, 8, 13, 3,  6, 0)
    #puzzle = generate_x_moves_from_goal(n, 25)
    print("Generated puzzle:", puzzle)

    heuristics = [
        (manhattan_distance, "Manhattan"),
        #(misplaced_tiles, "Misplaced"),
        (manhattan_linear_conflict, "ManhattanLinear")
    ]

    print("\nSolving the puzzle using A* algorithm with Manhattan distance heuristic...")
    start_time = time.time()
    for heuristic, name in heuristics:
        print(f"\nUsing {name} heuristic:")
        solution_path, steps, visited_states = astar(puzzle, heuristic, solved_puzzle)
        end_time = time.time()

        elapsed_time = end_time - start_time        

        if solution_path:
            print("\nSolution found!")
            print("Moves to solve the puzzle:", solution_path)
            print("Number of moves:", steps)
            print("Number of visited states:", visited_states)
            print("Time taken (seconds):", elapsed_time)
        else:
            print("\nNo solution found.")
    

# ========================== MAIN ==========================

if __name__ == "__main__":
    #run_heuristic_comparison(4)
    #compare_steps_visited_and_time(3)
    normal_test(4)