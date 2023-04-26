use std::time::Instant;

fn solve_queens(size: usize) -> Vec<Vec<usize>> {
    let mut result = vec![];
    let mut positions = vec![0; size];

    fn backtrack(
        size: usize,
        positions: &mut Vec<usize>,
        row: usize,
        diagonals: &mut [bool],
        anti_diagonals: &mut [bool],
        columns: &mut [bool],
        result: &mut Vec<Vec<usize>>,
    ) {
        if row == size {
            result.push(positions.clone());
            return;
        }

        for col in 0..size {
            let diagonal_index = (size - 1) + (row - col);
            let anti_diagonal_index = row + col;

            if columns[col] || diagonals[diagonal_index] || anti_diagonals[anti_diagonal_index] {
                continue;
            }

            positions[row] = col;
            columns[col] = true;
            diagonals[diagonal_index] = true;
            anti_diagonals[anti_diagonal_index] = true;

            backtrack(size, positions, row + 1, diagonals, anti_diagonals, columns, result);

            positions[row] = 0;
            columns[col] = false;
            diagonals[diagonal_index] = false;
            anti_diagonals[anti_diagonal_index] = false;
        }
    }

    let mut diagonals = vec![false; size * 2 - 1];
    let mut anti_diagonals = vec![false; size * 2 - 1];
    let mut columns = vec![false; size];

    backtrack(size, &mut positions, 0, &mut diagonals, &mut anti_diagonals, &mut columns, &mut result);

    result
}

fn main() {
    let size = 8;
    let start_time = Instant::now();
    let solutions = solve_queens(size);
    let count=solutions.len();
    let elapsed_time = start_time.elapsed();
    println!("Found {} solutions for {} queens in {:?}",count, size, elapsed_time);
    /*
    println!("Found {} solutions for {} queens:", solutions.len(), size);
    for solution in solutions {
        println!("{:?}", solution);
    }
    */
}

