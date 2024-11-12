class Sparsemax:
    def __init__(self):
        pass

    def call(self, logits):
        # Sort logits in descending order
        logits_sorted = sorted(logits, reverse=True)

        # Compute the cumulative sum of the sorted logits
        cumsum_logits = []
        current_sum = 0
        for logit in logits_sorted:
            current_sum += logit
            cumsum_logits.append(current_sum)

        # Compute threshold
        k_array = list(range(1, len(logits) + 1))
        threshold = [(cumsum_logits[i] - 1) / k_array[i] for i in range(len(logits))]

        # Find valid entries where logits exceed the threshold
        valid_entries = [i for i, l in enumerate(logits_sorted) if l > threshold[i]]

        # Determine k (number of valid entries)
        k = len(valid_entries)

        # Calculate tau (threshold to apply sparsity)
        tau = (cumsum_logits[k - 1] - 1) / k

        # Apply Sparsemax by zeroing out values below the threshold
        sparsemax_output = [max(logit - tau, 0) for logit in logits]

        return sparsemax_output

# Example usage of Sparsemax
def run_sparsemax():
    logits = [2.0, 1.0, 0.1, -1.0, -2.0]
    sparsemax = Sparsemax()
    output = sparsemax.call(logits)
    print("Sparsemax Output:", output)

run_sparsemax()
