using Flux
using JLD2
using Statistics

function main()
    println(" Loading data...")
    data = load("preprocessed_data.jld2")
    X = data["X"]
    y = data["y"]

    println("X shape before transpose: ", size(X))
    X = X'  # Shape: (sequence length, batch size)
    println("X shape after transpose: ", size(X))
    println("y shape: ", size(y))

    if any(X .<= 0)
        println(" Found zeros or negatives in X! Shifting indices up by 1.")
        X .= X .+ 1
    end

    vocab_size = maximum(X)
    embedding_dim = 8
    hidden_dim = 16

    model = Chain(
        Embedding(vocab_size, embedding_dim),
        x -> mean(x, dims=2),
        x -> reshape(x, embedding_dim, size(x,3)),
        Dense(embedding_dim, hidden_dim, relu),
        Dense(hidden_dim, 1),
        σ,  # sigmoid activation for binarycrossentropy
        x -> reshape(x, :)
    )

    println("Vocabulary size: $vocab_size")

    y = Float32.(y)
    @assert size(X,2) == length(y)

    opt = Flux.Adam()
    state = Flux.setup(opt, model)   # <-- NEW: setup optimizer state

    println(" Starting training...")
    epochs = 10
    for epoch in 1:epochs
        function loss(m)
            Flux.Losses.binarycrossentropy(m(X), y)
        end
        grads = gradient(loss, model)
        Flux.update!(state, model, grads)  # <-- NEW: use state
        l = loss(model)
        println("Epoch $epoch/$epochs | Loss = $l")
    end

    println(" Training complete.")
end

main()
