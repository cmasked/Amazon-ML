import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from rapidfuzz import fuzz
print('rapidfuzz test:')
r1 = fuzz.ratio("ABC Corp", "ABC Corporation")
r2 = fuzz.partial_ratio("ABC", "ABC Corporation")
r3 = fuzz.token_sort_ratio("Corp ABC", "ABC Corporation")
r4 = fuzz.token_set_ratio("Corp ABC Ltd", "ABC Corporation")
print(f"  ratio: {r1}")
print(f"  partial_ratio: {r2}")
print(f"  token_sort_ratio: {r3}")
print(f"  token_set_ratio: {r4}")
print("rapidfuzz OK")
