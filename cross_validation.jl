using Flux
using Random
using Statistics
using JLD2

println("🔹 Loading preprocessed data...")
data = JLD2.load("preprocessed_data.jld2")
X_raw = data["X"]
y_raw = Float32.(data["y"])

# Transpose to (sequence length, N)
X = X_raw'
y = y_raw

# Shift indices to be >=1
if any(X .<= 0)
    println("⚠️ Found zeros or negatives in X! Shifting indices up by 1.")
    X .= X .+ 1
end

N = size(X, 2)
println("✅ Dataset size: $N examples")

k = 10  # Number of folds

# Shuffle indices
Random.seed!(1234)
shuffled_indices = shuffle(1:N)

# Partition indices safely
folds = [shuffled_indices[i:min(i+cld(N,k)-1, N)] for i in 1:cld(N,k):N]

# Store accuracy per fold
accuracies = Float64[]

# Fix vocab_size to the max index of the WHOLE dataset
vocab_size = maximum(X)

embedding_dim = 8
hidden_dim = 16
epochs = 10

println("🔹 Starting 10-fold cross-validation...")

for (fold_num, val_idx) in enumerate(folds)
    train_idx = setdiff(shuffled_indices, val_idx)

    X_train = X[:, train_idx]
    y_train = y[train_idx]
    X_val = X[:, val_idx]
    y_val = y[val_idx]

    println("🟢 Fold $fold_num: Training $(length(train_idx)) samples, Validation $(length(val_idx)) samples")

    # Build fresh model
    model = Chain(
        Embedding(vocab_size, embedding_dim),
        x -> mean(x, dims=2),
        x -> reshape(x, embedding_dim, size(x,3)),
        Dense(embedding_dim, hidden_dim, relu),
        Dense(hidden_dim, 1),
        σ,
        x -> reshape(x, :)
    )

    opt = Flux.Adam()
    state = Flux.setup(opt, model)

    # Train
    for epoch in 1:epochs
      lossfun(m) = Flux.Losses.binarycrossentropy(m(X_train), y_train)
gs = gradient(lossfun, model)
Flux.update!(state, model, gs[1])

    end

    # Evaluate
    preds = model(X_val)
    preds_binary = round.(preds)
    acc = mean(preds_binary .== y_val)
    push!(accuracies, acc)

    println("✅ Fold $fold_num Validation Accuracy: $(round(acc*100, digits=2))%")
end

avg_acc = mean(accuracies)
println("🏁 10-Fold Cross-Validation Average Accuracy: $(round(avg_acc*100, digits=2))%")
