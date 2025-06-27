# test.py
import collections
import multiprocessing
import typing as t

# Assuming these are defined elsewhere or passed in
# For demonstration, let's provide dummy implementations
def get_pair_counts(tokens: t.List[int], p_counts: t.Dict[tuple[int, int], int]):
    """
    Counts pairs of adjacent tokens in a list and updates the given p_counts dictionary.
    This function should be designed to update a shared dictionary safely (though here it returns updates).
    For multiprocessing, it's safer for each process to return its own counts, then combine.
    """
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        p_counts[pair] = p_counts.get(pair, 0) + 1
    return p_counts # In multiprocessing, each worker returns its own dict

def merge_pair(tokens: t.List[int], pair_to_merge: tuple[int, int], new_token_id: int) -> t.List[int]:
    """
    Merges occurrences of a specific pair of tokens into a new token ID.
    """
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == pair_to_merge:
            merged_tokens.append(new_token_id)
            i += 2 # Skip both tokens that were merged
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens

# --- Your main class/function where the loop resides ---

class BPEProcessor:
    def __init__(self, num_merges: int, num_processes: int = None):
        self._num_merges = num_merges
        # Use CPU count if num_processes is not specified
        self._num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()

    def perform_merges(self, initial_chunks_tokens: t.List[t.List[int]], initial_vocab: t.Dict[int, bytes]):
        """
        Performs BPE merges with multiprocessing acceleration.
        """
        chunks_tokens = initial_chunks_tokens
        vocab = initial_vocab
        merge_ranks: t.Dict[tuple[int, int], int] = {} # Stores merge rules (pair -> new_token_id)

        print(f"Starting BPE merges with {self._num_processes} processes...")

        # Create a multiprocessing pool
        # It's good practice to create the pool outside the loop if it's long-running
        # and you want to reuse processes.
        # However, for a fixed number of merges, creating/closing inside the loop might be simpler
        # if the pool is not expected to be reused extensively across different high-level tasks.
        # For this specific scenario (looping through _num_merges), let's create it once outside.
        with multiprocessing.Pool(processes=self._num_processes) as pool:
            for i in range(self._num_merges):
                print(f"\n--- Merge Iteration {i+1}/{self._num_merges} ---")

                # --- Accelerate 1: Accumulate Pair Counts ---
                # Each process will call get_pair_counts on a chunk of tokens and return its local counts.
                # pool.map or pool.starmap are good for this.
                # Here, we'll map a helper that wraps get_pair_counts.
                
                # Each process will receive one 'tokens' list from chunks_tokens
                # and return its own dictionary of pair counts for that list.
                all_partial_p_counts = pool.map(
                    lambda tokens_list: get_pair_counts(tokens_list, {}), # Pass an empty dict for each process
                    chunks_tokens
                )
                
                # Manually accumulate all partial p_counts from workers
                p_counts: t.Dict[tuple[int, int], int] = collections.defaultdict(int)
                for partial_counts in all_partial_p_counts:
                    for pair, count in partial_counts.items():
                        p_counts[pair] += count

                if not p_counts:
                    print("No more pairs to merge. Stopping early.")
                    break

                # From p_counts find occur-most pair of tokens (two IDs) as top_pair
                occur_most_pair: tuple[int, int] = max(p_counts, key=p_counts.get)
                new_token: int = i + 256 # Use merge rank as new token ID

                merge_ranks[occur_most_pair] = new_token # Record merge: rank as new token
                vocab[new_token] = vocab[occur_most_pair[0]] + vocab[occur_most_pair[1]] # Record new token corresponding bytes

                print(f"Merging pair {occur_most_pair} (representing '{vocab[occur_most_pair[0]].decode(errors='replace')}' + '{vocab[occur_most_pair[1]].decode(errors='replace')}') into new token ID {new_token}")

                # --- Accelerate 2: Update chunks_tokens ---
                # Each process will call merge_pair on a chunk of tokens and return the updated chunk.
                # Use functools.partial to fix the pair_to_merge and new_token_id arguments.
                
                # Create a partially applied function for merge_pair
                from functools import partial
                merge_func_for_pool = partial(merge_pair,
                                              pair_to_merge=occur_most_pair,
                                              new_token_id=new_token)

                # Map this partial function over all token chunks
                chunks_tokens = pool.map(merge_func_for_pool, chunks_tokens)

        print("\nBPE merges completed.")
        return chunks_tokens, merge_ranks, vocab

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy initial data for demonstration
    # Initial tokens are ASCII values (b'a' -> 97, b'b' -> 98, etc.)
    # Let's say we start with some basic words
    initial_tokens_data = [
        [ord('a'), ord('b'), ord('a'), ord('b'), ord('a')], # ababa
        [ord('a'), ord('b'), ord('c')],                     # abc
        [ord('c'), ord('a'), ord('b')],                     # cab
        [ord('a'), ord('b'), ord('a'), ord('c')]            # abac
    ]

    # Initial vocab for base bytes (0-255 ASCII)
    initial_vocab_data = {i: bytes([i]) for i in range(256)}

    # Create an instance of the processor
    # For testing, you might use 2 processes, for production use multiprocessing.cpu_count()
    processor = BPEProcessor(num_merges=3, num_processes=2) # Perform 3 merges with 2 processes

    final_chunks, final_merge_ranks, final_vocab = processor.perform_merges(
        initial_tokens_data,
        initial_vocab_data
    )

    print("\n--- Final Results ---")
    print("Final Chunks Tokens:", final_chunks)
    print("Final Merge Ranks:", final_merge_ranks)
    print("Final Vocab:")
    for token_id, token_bytes in final_vocab.items():
        try:
            print(f"  {token_id}: '{token_bytes.decode('utf-8')}'")
        except UnicodeDecodeError:
            print(f"  {token_id}: {token_bytes} (undecodable)")


if __name__ == "__main__":
    tokens = []
    print(get_pair_counts(tokens))

    p_counts = {}
    returned = get_pair_counts(tokens, p_counts)

    print(returned)

    
