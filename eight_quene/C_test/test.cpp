#include <stdio.h>
#include <stdlib.h>

#define N 8

int board[N][N];

void print_solution() {
    static int count = 1;
    printf("Solution %d:\n", count++);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", board[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int is_safe(int row, int col) {
    int i, j;

    // Check this row on left side
    for (i = 0; i < col; i++) {
        if (board[row][i]) {
            return 0;
        }
    }

    // Check upper diagonal on left side
    for (i = row, j = col; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j]) {
            return 0;
        }
    }

    // Check lower diagonal on left side
    for (i = row, j = col; j >= 0 && i < N; i++, j--) {
        if (board[i][j]) {
            return 0;
        }
    }

    return 1;
}

void solve_queens(int col) {
    if (col == N) {
        print_solution();
        return;
    }

    for (int i = 0; i < N; i++) {
        if (is_safe(i, col)) {
            board[i][col] = 1;
            solve_queens(col + 1);
            board[i][col] = 0; // backtrack
        }
    }
}

int main() {
    solve_queens(0);
    return 0;
}

