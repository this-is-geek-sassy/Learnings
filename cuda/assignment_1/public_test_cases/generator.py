import numpy as np
import sys

def main():
    # Check if correct number of arguments provided
    if len(sys.argv) != 3:
        print("Usage: python program.py <rows> <cols>")
        print("Example: python program.py 10 20")
        sys.exit(1)
    
    try:
        # Parse command line arguments
        rows = int(sys.argv[1])
        cols = int(sys.argv[2])
        
        # Validate dimensions
        if rows <= 0 or cols <= 0:
            print("Error: Matrix dimensions must be positive integers")
            sys.exit(1)
            
    except ValueError:
        print("Error: Please provide valid integer values for rows and columns")
        sys.exit(1)
    
    print(f"Generating two random {rows}x{cols} matrices with int32 dtype...")
    
    # Set random seed for reproducibility (optional)
    np.random.seed(42)
    
    # Generate two random matrices with specified dimensions and int32 dtype
    # Using randint to generate integers in a reasonable range
    matrix1 = np.random.randint(-100, 101, size=(rows, cols), dtype=np.int32)
    matrix2 = np.random.randint(-100, 101, size=(rows, cols), dtype=np.int32)
    
    print(f"\nMatrix 1 ({rows}x{cols}):")
    print(matrix1)
    print(f"Shape: {matrix1.shape}, Data type: {matrix1.dtype}")
    
    print(f"\nMatrix 2 ({rows}x{cols}):")
    print(matrix2)
    print(f"Shape: {matrix2.shape}, Data type: {matrix2.dtype}")
    
    # Save matrices as CSV files
    np.savetxt('matrix1.csv', matrix1, delimiter=',', fmt='%d')
    np.savetxt('matrix2.csv', matrix2, delimiter=',', fmt='%d')
    print(f"\nMatrices saved as 'matrix1.csv' and 'matrix2.csv'")

if __name__ == "__main__":
    main()