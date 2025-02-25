def rowcol_to_colrow(tuple_list):
    """
    Switch (row, col) tuples to (col, row).

    Args:
        tuple_list (list): List of (row, col) tuples.
    
    Returns:
        list: List of (col, row) tuples.

    # Example usage
    tuple_list = [(1, 2), (3, 4), (5, 6)]
    switched = switch_row_col(tuple_list)
    print(switched)  # Output: [(2, 1), (4, 3), (6, 5)]
    
    """
    return [(col, row) for row, col in tuple_list]

    


