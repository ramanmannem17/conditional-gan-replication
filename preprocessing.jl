using CSV
using DataFrames

# Load dataset
df = CSV.read("election_sample.csv", DataFrame)

# Verify columns
@assert "tweet" in names(df) "Missing 'tweet' column"
@assert "label" in names(df) "Missing 'label' column"

# Tokenize tweets by splitting on spaces
tokenized = [split(lowercase(String(t))) for t in df.tweet]

# Build vocabulary
vocab = String[]
for tokens in tokenized
    append!(vocab, tokens)
end
vocab = unique(vocab)
vocab_dict = Dict(word => idx for (idx, word) in enumerate(vocab))

# Encode tweets as integer sequences
maxlen = maximum(length.(tokenized))
X = zeros(Int, length(tokenized), maxlen)
for (i, tokens) in enumerate(tokenized)
    for (j, token) in enumerate(tokens)
        X[i, j] = vocab_dict[token]
    end
end

# Encode labels: positive=1, negative=0
y = [l == "positive" ? 1 : 0 for l in df.label]

# Save processed arrays for later
using JLD2
@save "preprocessed_data.jld2" X y vocab_dict maxlen

println(" Preprocessing done. Saved to preprocessed_data.jld2")
