"""
Memory-efficient dataset profiling. 
Streams files instead of loading everything into pandas.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def profile_source_file(path, name):
    """Stream through a source file and collect statistics."""
    print(f"\n=== {name} ({os.path.basename(path)}) ===")
    countries = Counter()
    name_empty = 0
    addr_empty = 0
    name_na = 0
    addr_na = 0
    name_lengths = []
    addr_lengths = []
    total = 0
    
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        print(f"  Columns: {header}")
        
        for line in f:
            total += 1
            parts = line.strip().split('\t')
            
            # entity_id, business_name, business_address, country
            eid = parts[0] if len(parts) > 0 else ''
            bname = parts[1] if len(parts) > 1 else ''
            baddr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''
            
            countries[country] += 1
            
            if not bname.strip():
                name_empty += 1
            if not baddr.strip():
                addr_empty += 1
                
            name_lengths.append(len(bname))
            addr_lengths.append(len(baddr))
            
            if total <= 5:
                print(f"  Sample {total}: id={eid[:20]}, name={bname[:50]}, addr={baddr[:50]}, country={country}")
    
    print(f"  Total records: {total:,}")
    print(f"  Empty names: {name_empty:,} ({100*name_empty/max(total,1):.2f}%)")
    print(f"  Empty addresses: {addr_empty:,} ({100*addr_empty/max(total,1):.2f}%)")
    
    # Compute length stats from sampled data
    import numpy as np
    nl = np.array(name_lengths[:100000])  # Use first 100K for stats
    al = np.array(addr_lengths[:100000])
    print(f"  Name length: mean={nl.mean():.1f}, median={np.median(nl):.0f}, max={nl.max()}")
    print(f"  Addr length: mean={al.mean():.1f}, median={np.median(al):.0f}, max={al.max()}")
    print(f"  Countries: {dict(countries)}")
    
    return total, countries


def profile_ground_truth(path):
    """Profile the ground truth file."""
    print(f"\n=== GROUND TRUTH ===")
    
    total = 0
    singletons = 0
    match_counts = Counter()
    s2_total = 0
    s3_total = 0
    total_matches = 0
    
    with open(path, encoding='utf-8') as f:
        header = f.readline()
        
        for line in f:
            total += 1
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matched = parts[1] if len(parts) > 1 else ''
            
            if not matched.strip():
                singletons += 1
                match_counts[0] += 1
            else:
                ids = matched.split(',')
                n = len(ids)
                match_counts[n] += 1
                total_matches += n
                
                for mid in ids:
                    if mid.startswith('S2-'):
                        s2_total += 1
                    elif mid.startswith('S3-'):
                        s3_total += 1
    
    print(f"  Total S1 entities: {total:,}")
    print(f"  Singletons (0 matches): {singletons:,} ({100*singletons/total:.1f}%)")
    print(f"  With matches: {total - singletons:,} ({100*(total-singletons)/total:.1f}%)")
    print(f"  Total match IDs: {total_matches:,}")
    print(f"  S2 match IDs: {s2_total:,}")
    print(f"  S3 match IDs: {s3_total:,}")
    print(f"  Avg matches per entity (all): {total_matches/total:.3f}")
    print(f"  Avg matches per matched entity: {total_matches/max(total-singletons,1):.3f}")
    
    print(f"\n  Match count distribution:")
    for n in sorted(match_counts.keys()):
        if n <= 20 or match_counts[n] > 100:
            print(f"    {n} matches: {match_counts[n]:,} ({100*match_counts[n]/total:.2f}%)")
    
    return total, singletons, total_matches


# Run profiling
print("=" * 60)
print("DATASET PROFILING (streaming, memory-efficient)")
print("=" * 60)

t0 = time.time()

profile_source_file(os.path.join(BASE, 'dataset/train/train_source1.tsv'), 'Train Source 1')
profile_source_file(os.path.join(BASE, 'dataset/train/train_source2.tsv'), 'Train Source 2')
profile_source_file(os.path.join(BASE, 'dataset/train/train_source3.tsv'), 'Train Source 3')

profile_ground_truth(os.path.join(BASE, 'dataset/train/train_ground_truth.tsv'))

profile_source_file(os.path.join(BASE, 'dataset/test/test_source1.tsv'), 'Test Source 1')
profile_source_file(os.path.join(BASE, 'dataset/test/test_source2.tsv'), 'Test Source 2')
profile_source_file(os.path.join(BASE, 'dataset/test/test_source3.tsv'), 'Test Source 3')

print(f"\n\nTotal profiling time: {time.time()-t0:.0f}s")
