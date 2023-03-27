fn print_board(board: &Vec<Vec<u8>>) {
    for i in 0..9 {
        for j in 0..9 {
            print!("{:?} ", board[i][j]);
        }
        print!("\n");
    }
}

fn is_valid(board: &Vec<Vec<u8>>, row: usize, col: usize, num: u8) -> bool {
    for i in 0..9 {
        if board[row][i] == num || board[i][col] == num {
            return false;
        }
    }

    let sub_grid_row = (row / 3) * 3;
    let sub_grid_col = (col / 3) * 3;
    for i in 0..3 {
        for j in 0..3 {
            if board[sub_grid_row + i][sub_grid_col + j] == num {
                return false;
            }
        }
    }

    true
}

fn solve_sudoku(board: &mut Vec<Vec<u8>>, row: usize, col: usize) -> bool {
    if row == 9 {
        return true;
    }

    let next_row = if col == 8 { row + 1 } else { row };
    let next_col = if col == 8 { 0 } else { col + 1 };

    if board[row][col] > 0 {
        return solve_sudoku(board, next_row, next_col);
    }

    for i in 1..10 {
        if is_valid(board, row, col, i) {
            board[row][col] = i;
            if solve_sudoku(board, next_row, next_col) {
                return true;
            }
            board[row][col] = 0;
        }
    }

    false
}

fn main() {
    let mut board: Vec<Vec<u8>> = vec![
        vec![5, 3, 0, 0, 7, 0, 0, 0, 0],
        vec![6, 0, 0, 1, 9, 5, 0, 0, 0],
        vec![0, 0, 8, 0, 0, 0, 0, 6, 0],
        vec![8, 0, 0, 0, 6, 0, 0, 0, 3],
        vec![0, 0, 0, 0, 0, 3, 0, 0, 1],
        vec![7, 0, 0, 0, 2, 0, 0, 0, 6],
        vec![0, 6, 0, 0, 0, 0, 2, 8, 0],
        vec![0, 0, 0, 4, 1, 9, 0, 0, 5],
        vec![0, 4, 0, 0, 8, 0, 0, 0, 9],
    ];

    if solve_sudoku(&mut board, 0, 0) {
        print_board(&board);
    } else {
        println!("No solution found");
    }
}
