from docx import Document
import sys

def print_table(table):
    for row in table.rows:
        print(' | '.join(cell.text.strip() or ' ' for cell in row.cells))
    print('-' * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tables.py <file.docx>")
        sys.exit(1)
    doc = Document(sys.argv[1])
    for idx, table in enumerate(doc.tables):
        print(f"Table {idx+1}:")
        print_table(table)